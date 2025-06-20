#!/usr/bin/env python3
"""
SSL Certificate Generator for BEM System Production Deployment
Creates self-signed certificates for HTTPS/WSS support
"""

import subprocess
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_ssl_certificates():
    """Generate self-signed SSL certificates for production"""
    logger.info("🔐 Generating SSL certificates for production deployment...")
    
    try:
        # Generate private key
        logger.info("📝 Generating private key...")
        subprocess.run([
            "openssl", "genrsa", "-out", "key.pem", "2048"
        ], check=True, capture_output=True)
        
        # Generate certificate signing request
        logger.info("📋 Creating certificate signing request...")
        subprocess.run([
            "openssl", "req", "-new", "-key", "key.pem", "-out", "cert.csr",
            "-subj", "/C=US/ST=CA/L=SF/O=BEM/OU=Production/CN=localhost"
        ], check=True, capture_output=True)
        
        # Generate self-signed certificate
        logger.info("🎫 Generating self-signed certificate...")
        subprocess.run([
            "openssl", "x509", "-req", "-days", "365", "-in", "cert.csr",
            "-signkey", "key.pem", "-out", "cert.pem"
        ], check=True, capture_output=True)
        
        # Clean up CSR file
        os.remove("cert.csr")
        
        # Set appropriate permissions
        os.chmod("key.pem", 0o600)
        os.chmod("cert.pem", 0o644)
        
        logger.info("✅ SSL certificates generated successfully!")
        logger.info("📁 Files created: key.pem (private key), cert.pem (certificate)")
        logger.info("🔒 Production HTTPS/WSS ready")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to generate SSL certificates: {e}")
        logger.error("💡 Install OpenSSL: brew install openssl (macOS) or apt-get install openssl (Linux)")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    generate_ssl_certificates() 