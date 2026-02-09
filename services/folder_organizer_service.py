"""
Folder Organizer Service - organizes images into person-based folders.

This service:
1. Creates folder structure based on people in images
2. Copies/moves images to appropriate folders
3. Appends file size to filenames (e.g., _4_2MB.jpg)
4. Handles single person, multiple people, and group photos
"""
import logging
from typing import List, Dict, Set
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class FolderOrganizerService:
    """
    Organizes images into folders based on detected people.
    
    Folder naming:
    - Single person: 001/
    - Two people: 001_002/
    - Three people: 001_002_003/
    - More than 3: G001/, G002/, etc.
    
    File naming:
    - Appends file size: IMG_001_4_2MB.jpg (4.2 MB file)
    """
    
    def __init__(
        self,
        output_directory: Path,
        max_persons_named: int = 3,
        group_prefix: str = "G",
        copy_images: bool = True,
        append_file_size: bool = True
    ):
        """
        Initialize folder organizer.
        
        Args:
            output_directory: Root directory for organized folders
            max_persons_named: Maximum persons before using group naming
            group_prefix: Prefix for group folders
            copy_images: If True, copy images; if False, move images
            append_file_size: If True, append file size to filename
        """
        self.output_directory = Path(output_directory)
        self.max_persons_named = max_persons_named
        self.group_prefix = group_prefix
        self.copy_images = copy_images
        self.append_file_size = append_file_size
        self.group_counter = 1
        
        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"FolderOrganizerService initialized: "
            f"output={output_directory}, "
            f"copy_mode={copy_images}, "
            f"append_size={append_file_size}"
        )
    
    def organize_image(
        self,
        source_path: Path,
        person_ids: List[int],
        person_names: Dict[int, str]
    ) -> Path:
        """
        Organize a single image into appropriate folder.
        
        Args:
            source_path: Source image path
            person_ids: List of person IDs in the image (sorted)
            person_names: Dictionary mapping person_id → display_name
            
        Returns:
            Destination path where image was copied/moved
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Source image not found: {source_path}")
        
        if not person_ids:
            logger.warning(f"No persons for image {source_path.name}, skipping")
            return source_path
        
        # Sort person IDs for consistent naming
        person_ids = sorted(person_ids)
        
        # Determine folder name
        folder_name = self._get_folder_name(person_ids, person_names)
        
        # Create destination folder
        dest_folder = self.output_directory / folder_name
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate destination filename with file size
        dest_filename = self._get_dest_filename(source_path)
        dest_path = dest_folder / dest_filename
        
        # Handle filename conflicts
        dest_path = self._resolve_conflict(dest_path)
        
        # Copy or move file
        if self.copy_images:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied: {source_path.name} -> {folder_name}/{dest_path.name}")
        else:
            shutil.move(str(source_path), dest_path)
            logger.info(f"Moved: {source_path.name} -> {folder_name}/{dest_path.name}")
        
        return dest_path
    
    def _get_folder_name(self, person_ids: List[int], person_names: Dict[int, str]) -> str:
        """
        Generate folder name based on person IDs.
        
        Args:
            person_ids: Sorted list of person IDs
            person_names: Person name mapping
            
        Returns:
            Folder name (e.g., "001", "001_002", "G001")
        """
        num_persons = len(person_ids)
        
        if num_persons <= self.max_persons_named:
            # Use person names joined with underscore
            names = [person_names.get(pid, f"{pid:03d}") for pid in person_ids]
            return "_".join(names)
        else:
            # Use group naming
            folder_name = f"{self.group_prefix}{self.group_counter:03d}"
            self.group_counter += 1
            return folder_name
    
    def _get_dest_filename(self, source_path: Path) -> str:
        """
        Generate destination filename with optional file size suffix.
        
        Args:
            source_path: Source file path
            
        Returns:
            Filename with size suffix (e.g., "IMG_001_4_2MB.jpg")
        """
        if not self.append_file_size:
            return source_path.name
        
        # Get file size
        file_size_bytes = source_path.stat().st_size
        size_suffix = self._format_file_size(file_size_bytes)
        
        # Split filename and extension
        stem = source_path.stem  # filename without extension
        ext = source_path.suffix  # .jpg, .png, etc.
        
        # Append size suffix
        new_filename = f"{stem}_{size_suffix}{ext}"
        
        return new_filename
    
    def _format_file_size(self, size_bytes: int) -> str:
        """
        Format file size for filename suffix.
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Formatted size (e.g., "4_2MB", "850KB")
        """
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024
        
        if size_mb >= 1.0:
            # Format as MB with 1 decimal, replace . with _
            mb_str = f"{size_mb:.1f}".replace(".", "_")
            return f"{mb_str}MB"
        else:
            # Format as KB (whole number)
            return f"{int(size_kb)}KB"
    
    def _resolve_conflict(self, dest_path: Path) -> Path:
        """
        Resolve filename conflicts by adding counter.
        
        Args:
            dest_path: Proposed destination path
            
        Returns:
            Available path (may have counter appended)
        """
        if not dest_path.exists():
            return dest_path
        
        # File exists, add counter
        stem = dest_path.stem
        ext = dest_path.suffix
        parent = dest_path.parent
        
        counter = 1
        while True:
            new_name = f"{stem}_{counter}{ext}"
            new_path = parent / new_name
            
            if not new_path.exists():
                return new_path
            
            counter += 1
            
            if counter > 100:
                raise RuntimeError(f"Too many conflicts for {dest_path.name}")
    
    def organize_batch(
        self,
        image_person_mapping: Dict[Path, List[int]],
        person_names: Dict[int, str]
    ) -> Dict[Path, Path]:
        """
        Organize multiple images at once.
        
        Args:
            image_person_mapping: Dictionary mapping image_path → list of person_ids
            person_names: Dictionary mapping person_id → display_name
            
        Returns:
            Dictionary mapping source_path → destination_path
        """
        results = {}
        
        for source_path, person_ids in image_person_mapping.items():
            try:
                dest_path = self.organize_image(source_path, person_ids, person_names)
                results[source_path] = dest_path
            except Exception as e:
                logger.error(f"Failed to organize {source_path.name}: {e}")
        
        logger.info(f"Organized {len(results)}/{len(image_person_mapping)} images")
        
        return results
    
    def get_folder_summary(self) -> Dict[str, int]:
        """
        Get summary of organized folders.
        
        Returns:
            Dictionary mapping folder_name → file_count
        """
        summary = {}
        
        for folder in self.output_directory.iterdir():
            if folder.is_dir():
                file_count = len(list(folder.glob("*.*")))
                summary[folder.name] = file_count
        
        return summary
