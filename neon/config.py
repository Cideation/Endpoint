#!/usr/bin/env python3
"""
Configuration file for Neon PostgreSQL integration
"""

import os

# Neon PostgreSQL Configuration
NEON_CONFIG = {
    'host': 'ep-white-waterfall-a85g0dgx-pooler.eastus2.azure.neon.tech',
    'port': 5432,
    'database': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_CcgA0kKeYVU2',
    'sslmode': 'require'
}

# Connection string
DATABASE_URL = 'postgresql://neondb_owner:npg_CcgA0kKeYVU2@ep-white-waterfall-a85g0dgx-pooler.eastus2.azure.neon.tech/neondb?sslmode=require'

# Connection Pool Settings
POOL_MIN_SIZE = 5
POOL_MAX_SIZE = 20

# Stack Auth (if needed for future integration)
STACK_CONFIG = {
    'project_id': 'ed81c2da-23cd-4dd4-9817-dcdb3b989f13',
    'publishable_key': 'pck_5hn6ahmcsdjbkkfaepewy365th46hx54hbdk7ctyj8zk8',
    'secret_key': 'ssk_ew9w532ae56e4rc88v4c8jnjv7bq65k5xgcxn8zresxp8'
} 