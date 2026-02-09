"""
Person Service - manages person identification and naming.
"""
import logging
from typing import List, Dict, Optional
from sqlalchemy.orm import Session

from infrastructure.database.repositories import PersonRepository, ClusterRepository

logger = logging.getLogger(__name__)


class PersonService:
    """
    Service for managing person entities and IDs.
    
    Responsibilities:
    - Assign numeric IDs to persons (001, 002, etc.)
    - Format person names according to config
    - Handle person renaming
    """
    
    def __init__(self, person_id_format: str = "{:03d}"):
        """
        Initialize person service.
        
        Args:
            person_id_format: Format string for person IDs (default: {:03d} → 001, 002, etc.)
        """
        self.person_id_format = person_id_format
        logger.info(f"PersonService initialized with format: {person_id_format}")
    
    def create_person(self, session: Session) -> int:
        """
        Create a new person with auto-generated ID.
        
        Args:
            session: Database session
            
        Returns:
            Person ID
        """
        person_repo = PersonRepository(session)
        
        # Get next ID
        next_id = person_repo.get_next_id()
        
        # Format display name
        display_name = self.person_id_format.format(next_id)
        
        # Create person
        person = person_repo.create(display_name=display_name)
        
        return person.person_id
    
    def get_person_name(self, session: Session, person_id: int) -> Optional[str]:
        """
        Get person's display name.
        
        Args:
            session: Database session
            person_id: Person ID
            
        Returns:
            Display name or None
        """
        person_repo = PersonRepository(session)
        person = person_repo.get_by_id(person_id)
        
        return person.display_name if person else None
    
    def rename_person(self, session: Session, person_id: int, new_name: str) -> bool:
        """
        Rename a person.
        
        Args:
            session: Database session
            person_id: Person ID
            new_name: New display name
            
        Returns:
            True if successful
        """
        person_repo = PersonRepository(session)
        return person_repo.update_name(person_id, new_name)
    
    def get_all_persons(self, session: Session) -> List[Dict]:
        """
        Get all persons with their cluster counts.
        
        Args:
            session: Database session
            
        Returns:
            List of person dictionaries
        """
        person_repo = PersonRepository(session)
        cluster_repo = ClusterRepository(session)
        
        persons = person_repo.get_all()
        
        result = []
        for person in persons:
            clusters = cluster_repo.get_by_person(person.person_id)
            
            result.append({
                'person_id': person.person_id,
                'display_name': person.display_name,
                'cluster_count': len(clusters),
                'created_at': person.created_at
            })
        
        return result
