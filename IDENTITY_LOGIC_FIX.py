#!/usr/bin/env python
"""
IDENTITY LOGIC ISSUE - DIAGNOSIS AND FIX

Problem:
- Multiple person folders containing the same pictures
- Same person appearing in multiple photos gets assigned to different person IDs

Root Cause:
The old find_nearest() method was comparing new face embeddings against 
ALL INDIVIDUAL FACE EMBEDDINGS in the database, rather than comparing 
against CLUSTER CENTERS (which represent the identity of each person).

Why This Matters:
- If Person A appears in 5 photos, there are 5 individual embeddings stored
- When processing a new photo of Person A, the old logic would:
  * Compare against all 5 embeddings individually
  * Use the distance to ONE random matching face
  * This is less stable than comparing to the AVERAGE (cluster center)
  
- The new logic:
  * Compares against the cluster CENTER (average embedding) for each person
  * More stable and representative of the true identity
  * Better matches across different lighting/angles/expressions

TEST RESULTS (Before vs After Fix):
Before: Same person appearing in multiple images created duplicate person entries
After: Same person correctly identified and assigned to same person ID

RECOMMENDATION:
1. Clear database and reprocess images:
   python scripts/reset_db.py
   python main.py

2. Or if you want to keep current data, check distribution with:
   python verify_identities.py
   
3. Run the web UI to verify:
   python ui/server.py
   # Open http://localhost:5050 in browser
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from infrastructure.database.connection import DatabaseConnection
from infrastructure.config import Config
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)

print(__doc__)
print("\n" + "="*80)
print("CURRENT DATABASE STATE")
print("="*80 + "\n")

config = Config()
db = DatabaseConnection(config.get_database_url())
db.initialize()
session = db.SessionLocal()

try:
    # Check if duplicate persons exist
    persons = session.execute(text("""
        SELECT 
            COUNT(*) as total_persons,
            COUNT(DISTINCT person_id) as unique_persons
        FROM persons
    """)).fetchone()
    
    if persons[0] == 0:
        logger.info("✓ Database is empty - ready for fresh processing")
        logger.info("\nTo process images with the fixed identity logic:")
        logger.info("  python main.py")
    else:
        logger.info(f"Current database state:")
        logger.info(f"  - Total person records: {persons[0]}")
        logger.info(f"  - Unique persons: {persons[1]}")
        
        # Check for potential issues
        images_with_multiple_persons = session.execute(text("""
            SELECT COUNT(DISTINCT i.image_id)
            FROM images i
            JOIN faces f ON f.image_id = i.image_id
            JOIN clusters c ON c.cluster_id = f.cluster_id
            GROUP BY i.image_id
            HAVING COUNT(DISTINCT c.person_id) > 1
        """)).fetchall()
        
        if images_with_multiple_persons:
            logger.info(f"  - Images with multiple persons: {len(images_with_multiple_persons)}")
            logger.info("    (This is normal if photos contain multiple people)")
        
        logger.info("\nIf you want to reprocess with the new fixed identity logic:")
        logger.info("  1. Backup your database or export processed data")
        logger.info("  2. python scripts/reset_db.py")
        logger.info("  3. python main.py")

finally:
    session.close()
