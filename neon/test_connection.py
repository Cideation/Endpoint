"""
Simple test script to verify Neon database connection
"""

import asyncio
from db_manager import NeonDBManager
from config import NEON_CONFIG

async def test_connection():
    """Test basic database connection"""
    print("Testing Neon Database Connection...")
    print(f"Host: {NEON_CONFIG['host']}")
    print(f"Database: {NEON_CONFIG['database']}")
    print(f"User: {NEON_CONFIG['user']}")
    
    db_manager = NeonDBManager(NEON_CONFIG)
    
    try:
        # Test connection pool
        await db_manager.create_pool()
        print("✅ Connection pool created successfully")
        
        # Test basic query
        result = await db_manager.execute_query("SELECT version()")
        print(f"✅ Database version: {result[0]['version']}")
        
        # Test table existence
        tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """
        tables = await db_manager.execute_query(tables_query)
        print(f"✅ Found {len(tables)} tables:")
        for table in tables:
            print(f"   - {table['table_name']}")
        
        # Test component count if table exists
        component_tables = [t['table_name'] for t in tables if 'component' in t['table_name'].lower()]
        if component_tables:
            print(f"\nComponent-related tables: {component_tables}")
            
            # Try to count components
            try:
                count_result = await db_manager.execute_query("SELECT COUNT(*) as count FROM components")
                component_count = count_result[0]['count']
                print(f"✅ Components in database: {component_count}")
            except Exception as e:
                print(f"⚠️  Could not count components: {e}")
        
        print("\n🎉 Database connection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    finally:
        await db_manager.close_async()

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    exit(0 if success else 1) 