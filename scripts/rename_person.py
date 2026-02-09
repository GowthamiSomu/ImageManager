"""
Rename person utility script.

Allows renaming person IDs (001, 002) to custom names (John, Mary).
Updates database and can optionally reorganize folders.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

from infrastructure.config import Config
from infrastructure.logging import setup_logging
from infrastructure.database.connection import DatabaseConnection
from infrastructure.database.repositories import PersonRepository

logger = logging.getLogger(__name__)


def main():
    """Rename a person."""
    parser = argparse.ArgumentParser(description='Rename a person in ImageManager database')
    parser.add_argument('--person-id', type=str, required=True, 
                        help='Person ID or display name (e.g., 001 or existing custom name)')
    parser.add_argument('--new-name', type=str, required=True, help='New name for the person')
    parser.add_argument('--reorganize', action='store_true', help='Reorganize folders after renaming')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Setup logging
    setup_logging(
        log_level='INFO',
        log_file=None
    )
    
    logger.info("=" * 60)
    logger.info("ImageManager - Person Rename Utility")
    logger.info("=" * 60)
    
    # Initialize database
    db = DatabaseConnection(config.get_database_url())
    db.initialize()
    
    # Perform rename
    with db.get_session() as session:
        person_repo = PersonRepository(session)
        
        # Try to find person by display name first, then by ID
        person = person_repo.get_by_name(args.person_id)
        
        if not person:
            # Try as numeric ID
            try:
                person_id_int = int(args.person_id)
                person = person_repo.get_by_id(person_id_int)
            except ValueError:
                pass
        
        if not person:
            logger.error(f"Person '{args.person_id}' not found")
            logger.info("\nUse 'python scripts/person_stats.py' to see all persons")
            sys.exit(1)
        
        old_name = person.display_name
        
        logger.info(f"Person ID: {person.person_id}")
        logger.info(f"Current name: {old_name}")
        logger.info(f"New name: {args.new_name}")
        logger.info("")
        
        # Confirm
        response = input("Proceed with rename? (yes/no): ")
        
        if response.lower() not in ['yes', 'y']:
            logger.info("Rename cancelled")
            sys.exit(0)
        
        # Update database
        success = person_repo.update_name(person.person_id, args.new_name)
        
        if success:
            logger.info("")
            logger.info(f"[OK] Person renamed: {old_name} -> {args.new_name}")
            
            if args.reorganize:
                logger.info("")
                logger.info("Note: Folder reorganization requires re-running main.py")
                logger.info("  1. Delete organized folders")
                logger.info("  2. Run: python main.py")
            
            logger.info("=" * 60)
        else:
            logger.error("Rename failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
