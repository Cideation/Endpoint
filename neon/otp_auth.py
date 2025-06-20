"""
SMS-based One-Time Password (OTP) Authentication Module
Supports multiple SMS providers (Twilio, etc.)
"""

import os
import time
import random
import logging
import phonenumbers
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SMSProvider(ABC):
    """Abstract base class for SMS providers"""
    
    @abstractmethod
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS message to phone number"""
        pass

class TwilioProvider(SMSProvider):
    """Twilio SMS provider implementation"""
    
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = os.getenv('TWILIO_FROM_NUMBER')
        
        if not all([self.account_sid, self.auth_token, self.from_number]):
            raise ValueError("Missing required Twilio environment variables")
        
        self.client = Client(self.account_sid, self.auth_token)
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS using Twilio"""
        try:
            self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=phone_number
            )
            return True
        except TwilioRestException as e:
            logger.error(f"Failed to send SMS via Twilio: {str(e)}")
            return False

class OTPManager:
    """Manages OTP generation, verification, and SMS delivery"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize OTP manager with Redis for OTP storage"""
        self.redis_client = redis.from_url(redis_url)
        self.sms_provider = TwilioProvider()  # Can be swapped with other providers
        
        # OTP configuration
        self.OTP_LENGTH = 6
        self.OTP_EXPIRY = 300  # 5 minutes
        self.MAX_ATTEMPTS = 3
        self.COOLDOWN_PERIOD = 60  # 1 minute between retries
    
    def _generate_otp(self) -> str:
        """Generate a random OTP code"""
        return ''.join(random.choices('0123456789', k=self.OTP_LENGTH))
    
    def _validate_phone_number(self, phone_number: str) -> Tuple[bool, str]:
        """Validate and format phone number"""
        try:
            parsed = phonenumbers.parse(phone_number, None)
            if not phonenumbers.is_valid_number(parsed):
                return False, "Invalid phone number format"
            
            formatted = phonenumbers.format_number(
                parsed, 
                phonenumbers.PhoneNumberFormat.E164
            )
            return True, formatted
        except phonenumbers.NumberParseException:
            return False, "Could not parse phone number"
    
    def _get_rate_limit_key(self, phone_number: str) -> str:
        """Get rate limit key for phone number"""
        return f"otp:ratelimit:{phone_number}"
    
    def _get_otp_key(self, phone_number: str) -> str:
        """Get OTP storage key for phone number"""
        return f"otp:code:{phone_number}"
    
    def _get_attempts_key(self, phone_number: str) -> str:
        """Get attempts counter key for phone number"""
        return f"otp:attempts:{phone_number}"
    
    def can_request_otp(self, phone_number: str) -> Tuple[bool, str]:
        """Check if user can request a new OTP"""
        rate_limit_key = self._get_rate_limit_key(phone_number)
        
        # Check cooldown period
        if self.redis_client.exists(rate_limit_key):
            ttl = self.redis_client.ttl(rate_limit_key)
            return False, f"Please wait {ttl} seconds before requesting another code"
        
        return True, ""
    
    async def request_otp(self, phone_number: str) -> Tuple[bool, str]:
        """Generate and send OTP via SMS"""
        # Validate phone number
        is_valid, formatted_number = self._validate_phone_number(phone_number)
        if not is_valid:
            return False, formatted_number
        
        # Check rate limiting
        can_request, message = self.can_request_otp(formatted_number)
        if not can_request:
            return False, message
        
        # Generate and store OTP
        otp = self._generate_otp()
        otp_key = self._get_otp_key(formatted_number)
        rate_limit_key = self._get_rate_limit_key(formatted_number)
        
        # Store OTP with expiry
        pipe = self.redis_client.pipeline()
        pipe.setex(otp_key, self.OTP_EXPIRY, otp)
        pipe.setex(rate_limit_key, self.COOLDOWN_PERIOD, "1")
        pipe.execute()
        
        # Send OTP via SMS
        message = f"Your BEM System verification code is: {otp}. Valid for {self.OTP_EXPIRY//60} minutes."
        if not self.sms_provider.send_sms(formatted_number, message):
            return False, "Failed to send SMS. Please try again later."
        
        return True, "Verification code sent successfully"
    
    async def verify_otp(self, phone_number: str, otp: str) -> Tuple[bool, str, Optional[Dict]]:
        """Verify OTP code and return session data if valid"""
        # Validate phone number
        is_valid, formatted_number = self._validate_phone_number(phone_number)
        if not is_valid:
            return False, formatted_number, None
        
        otp_key = self._get_otp_key(formatted_number)
        attempts_key = self._get_attempts_key(formatted_number)
        
        # Check if OTP exists
        stored_otp = self.redis_client.get(otp_key)
        if not stored_otp:
            return False, "Verification code expired or invalid", None
        
        # Check attempts
        attempts = int(self.redis_client.get(attempts_key) or 0)
        if attempts >= self.MAX_ATTEMPTS:
            return False, "Too many failed attempts. Please request a new code.", None
        
        # Verify OTP
        if otp != stored_otp.decode():
            # Increment attempts counter
            pipe = self.redis_client.pipeline()
            pipe.incr(attempts_key)
            pipe.expire(attempts_key, self.OTP_EXPIRY)
            pipe.execute()
            
            remaining = self.MAX_ATTEMPTS - (attempts + 1)
            return False, f"Invalid code. {remaining} attempts remaining.", None
        
        # OTP verified - clean up
        pipe = self.redis_client.pipeline()
        pipe.delete(otp_key)
        pipe.delete(attempts_key)
        pipe.execute()
        
        # Return session data
        session_data = {
            'phone_number': formatted_number,
            'verified_at': int(time.time()),
            'auth_method': 'sms_otp'
        }
        
        return True, "Phone number verified successfully", session_data

# Example FastAPI integration:
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
otp_manager = OTPManager()

class PhoneNumber(BaseModel):
    phone_number: str

class OTPVerification(BaseModel):
    phone_number: str
    otp: str

@app.post("/auth/request-otp")
async def request_otp(phone: PhoneNumber):
    success, message = await otp_manager.request_otp(phone.phone_number)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}

@app.post("/auth/verify-otp")
async def verify_otp(verification: OTPVerification):
    success, message, session_data = await otp_manager.verify_otp(
        verification.phone_number,
        verification.otp
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {
        "message": message,
        "session": session_data
    }
""" 