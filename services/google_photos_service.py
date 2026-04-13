"""
Google Photos service for downloading and managing photos from Google Photos library.
Handles OAuth2 authentication, listing photos, and batch downloads.

OAuth Fix: Uses port 8085 for the local redirect server (not 8000 which conflicts
with other services). Ensure http://localhost:8085/ is added as an Authorized 
Redirect URI in your Google Cloud Console OAuth client settings.
"""
import logging
import os
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

# Google Photos API scopes
PHOTOS_SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']

# Required metadata fields for photos
MEDIA_METADATA_FIELDS = 'id,mediaMetadata,filename,productUrl,mimeType,baseUrl'

# Dedicated OAuth callback port — add http://localhost:8085/ to Google Cloud Console
# Authorized Redirect URIs. Do NOT use 8000 (conflicts with AI service).
OAUTH_PORT = 8085
OAUTH_FALLBACK_PORTS = [8085, 8086, 8087, 8088]


class GooglePhotosService:
    """Service for interacting with Google Photos API."""
    
    def __init__(
        self,
        credentials_file: str = None,
        client_id: str = None,
        client_secret: str = None,
        token_file: str = '.google_photos_token.json',
        output_dir: str = 'D:\\Photos'
    ):
        """
        Initialize Google Photos service.
        
        Args:
            credentials_file: Path to Google OAuth2 credentials JSON file
            client_id: Google OAuth2 Client ID (alternative to credentials_file)
            client_secret: Google OAuth2 Client Secret (alternative to credentials_file)
            token_file: Path to store/load OAuth2 token
            output_dir: Directory to download photos to
            
        IMPORTANT - Google Cloud Console setup:
            Add these Authorized Redirect URIs to your OAuth client:
              http://localhost:8085/
              http://localhost:8086/
              http://localhost:8087/
            Do NOT use http://localhost:8000/ — that port is used by the AI service.
        """
        self.credentials_file = credentials_file
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_file = token_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.service = None
        self.credentials = None
        self._authenticate()
        
    def _try_run_local_server(self, flow: InstalledAppFlow) -> Credentials:
        """
        Attempt OAuth flow on available ports.
        
        Tries OAUTH_FALLBACK_PORTS in order. This handles the case where
        a port is already in use (e.g. previous failed auth attempt).
        
        Args:
            flow: Configured InstalledAppFlow
            
        Returns:
            Credentials from completed OAuth flow
            
        Raises:
            RuntimeError: If all ports fail
        """
        last_error = None
        
        for port in OAUTH_FALLBACK_PORTS:
            try:
                logger.info(f"Starting OAuth server on http://localhost:{port}/")
                logger.info(f"  → Make sure http://localhost:{port}/ is in your")
                logger.info(f"    Google Cloud Console → OAuth client → Authorized redirect URIs")
                credentials = flow.run_local_server(
                    port=port,
                    prompt='consent',
                    # Open browser automatically
                    open_browser=True,
                    # Success message shown in browser
                    success_message='Authentication complete! You can close this tab and return to the terminal.'
                )
                logger.info(f"OAuth completed successfully on port {port}")
                return credentials
            except OSError as e:
                logger.warning(f"Port {port} unavailable: {e}, trying next…")
                last_error = e
            except Exception as e:
                # Non-port error (e.g. user cancelled) — propagate immediately
                raise
        
        raise RuntimeError(
            f"Could not start OAuth server on any of {OAUTH_FALLBACK_PORTS}. "
            f"Last error: {last_error}. "
            f"Free one of these ports and retry."
        )

    def _authenticate(self) -> None:
        """
        Authenticate with Google Photos API.
        
        Uses OAuth2 flow with token caching.
        Supports both credentials file and direct client_id/secret.
        
        Port used for local redirect: 8085 (not 8000).
        Add http://localhost:8085/ to Google Cloud Console Authorized Redirect URIs.
        """
        # Load existing token if available
        if os.path.exists(self.token_file):
            try:
                self.credentials = Credentials.from_authorized_user_file(
                    self.token_file, PHOTOS_SCOPES
                )
                logger.info(f"Loaded existing credentials from {self.token_file}")
            except Exception as e:
                logger.warning(f"Failed to load token: {e}. Will re-authenticate.")
                self.credentials = None
        
        # Refresh or create new credentials
        if not self.credentials or not self.credentials.valid:
            if (self.credentials and self.credentials.expired and 
                    self.credentials.refresh_token):
                # Refresh expired token
                try:
                    self.credentials.refresh(Request())
                    logger.info("Refreshed expired credentials")
                except Exception as e:
                    logger.error(f"Failed to refresh credentials: {e}")
                    self.credentials = None
            
            # If no valid credentials, run OAuth2 flow
            if not self.credentials:
                flow = None
                
                # Use credentials file if provided
                if self.credentials_file and os.path.exists(self.credentials_file):
                    logger.info(f"Using credentials file: {self.credentials_file}")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, PHOTOS_SCOPES
                    )
                # Use client_id and client_secret if provided
                elif self.client_id and self.client_secret:
                    logger.info("Using client_id and client_secret for authentication")
                    # IMPORTANT: redirect_uris must include all fallback ports
                    # These must ALSO be added in Google Cloud Console
                    client_config = {
                        "installed": {
                            "client_id": self.client_id,
                            "client_secret": self.client_secret,
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                            "redirect_uris": [
                                f"http://localhost:{p}/"
                                for p in OAUTH_FALLBACK_PORTS
                            ] + ["urn:ietf:wg:oauth:2.0:oob"],
                        }
                    }
                    flow = InstalledAppFlow.from_client_config(client_config, PHOTOS_SCOPES)
                else:
                    raise ValueError(
                        "Must provide either credentials_file or both client_id and client_secret.\n\n"
                        "Google Cloud Console setup required:\n"
                        "  1. Go to APIs & Services → Credentials → your OAuth 2.0 Client ID\n"
                        "  2. Under 'Authorized redirect URIs', add:\n"
                        f"       http://localhost:8085/\n"
                        f"       http://localhost:8086/\n"
                        f"       http://localhost:8087/\n"
                        "  3. Save and retry.\n"
                        "  (Remove http://localhost:8000/ if you added it — that port conflicts.)"
                    )
                
                self.credentials = self._try_run_local_server(flow)
                logger.info("Completed OAuth2 authentication")
        
        # Save credentials for future use
        with open(self.token_file, 'w') as f:
            f.write(self.credentials.to_json())
        
        # Build Google Photos API service
        self.service = build('photoslibrary', 'v1', credentials=self.credentials,
                             discoveryServiceUrl='https://photoslibrary.googleapis.com/$discovery/rest?version=v1')
        logger.info("Google Photos API service initialized")
    
    def list_photos(self, page_size: int = 100) -> List[Dict]:
        """
        List all photos in user's Google Photos library.
        
        Args:
            page_size: Number of photos per page (max 100)
            
        Returns:
            List of photo metadata dictionaries
        """
        all_photos = []
        next_page_token = None
        
        try:
            while True:
                logger.info(f"Fetching photos (page size: {page_size})")
                
                params = {
                    'pageSize': page_size,
                    'fields': MEDIA_METADATA_FIELDS
                }
                if next_page_token:
                    params['pageToken'] = next_page_token
                
                results = self.service.mediaItems().list(**params).execute()
                
                media_items = results.get('mediaItems', [])
                all_photos.extend(media_items)
                
                logger.info(f"  Retrieved {len(media_items)} photos (total: {len(all_photos)})")
                
                next_page_token = results.get('nextPageToken')
                if not next_page_token:
                    break
            
            logger.info(f"Total photos in library: {len(all_photos)}")
            return all_photos
            
        except Exception as e:
            logger.error(f"Failed to list photos: {e}")
            raise
    
    def download_photo(
        self,
        photo: Dict,
        preserve_metadata: bool = True
    ) -> Optional[Tuple[Path, Dict]]:
        """
        Download a single photo from Google Photos.
        
        Args:
            photo: Photo metadata dictionary from Google Photos API
            preserve_metadata: Whether to save photo metadata as JSON sidecar
            
        Returns:
            Tuple of (file_path, metadata) or None if download failed
        """
        try:
            photo_id = photo.get('id')
            filename = photo.get('filename', f'photo_{photo_id}.jpg')
            base_url = photo.get('baseUrl')
            media_metadata = photo.get('mediaMetadata', {})
            
            if not base_url:
                logger.warning(f"No download URL for photo {photo_id}")
                return None
            
            # =d for full-quality download, =dv for video
            mime_type = photo.get('mimeType', '')
            suffix = '=dv' if mime_type.startswith('video/') else '=d'
            download_url = f"{base_url}{suffix}"
            
            # Prepare output path
            output_path = self.output_dir / filename
            
            # Avoid duplicates
            counter = 1
            stem = output_path.stem
            ext_suffix = output_path.suffix
            while output_path.exists():
                output_path = self.output_dir / f"{stem}_{counter}{ext_suffix}"
                counter += 1
            
            # Download photo
            logger.info(f"Downloading: {filename}")
            response = requests.get(download_url, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Saved: {output_path.name} ({file_size_mb:.1f}MB)")
            
            # Save metadata as JSON sidecar
            if preserve_metadata:
                metadata_file = output_path.with_name(output_path.stem + '_metadata.json')
                metadata = {
                    'photo_id': photo_id,
                    'filename': filename,
                    'media_type': mime_type,
                    'created_time': media_metadata.get('creationTime'),
                    'width': media_metadata.get('width'),
                    'height': media_metadata.get('height'),
                    'camera_make': media_metadata.get('photo', {}).get('cameraMake'),
                    'camera_model': media_metadata.get('photo', {}).get('cameraModel'),
                    'focal_length': media_metadata.get('photo', {}).get('focalLength'),
                    'aperture': media_metadata.get('photo', {}).get('apertureFNumber'),
                    'iso': media_metadata.get('photo', {}).get('isoEquivalent'),
                    'shutter_speed': media_metadata.get('photo', {}).get('exposureTime'),
                }
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            return (output_path, photo)
            
        except Exception as e:
            logger.error(f"Failed to download photo {photo.get('id')}: {e}")
            return None
    
    def download_all_photos(
        self,
        max_photos: Optional[int] = None,
        batch_size: int = 10,
        preserve_metadata: bool = True,
        order_by: str = "creationTime"
    ) -> Dict[str, any]:
        """
        Download all photos from Google Photos library.
        
        Args:
            max_photos: Maximum number of photos to download (None = all)
            batch_size: Progress reporting interval
            preserve_metadata: Whether to save metadata for each photo
            order_by: Sort order - "creationTime" or "relevance"
            
        Returns:
            Dictionary with download statistics
        """
        logger.info("Fetching photo list from Google Photos...")
        photos = self.list_photos()
        
        if order_by == "creationTime":
            photos.sort(
                key=lambda p: p.get('mediaMetadata', {}).get('creationTime', ''),
                reverse=False
            )
            logger.info("Sorted by creation time (oldest first)")
        
        if max_photos:
            photos = photos[:max_photos]
            logger.info(f"Limited to {max_photos} photos")
        
        stats = {
            'total': len(photos),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_size_mb': 0,
            'start_time': datetime.now(),
            'downloaded_files': []
        }
        
        logger.info(f"Starting download of {stats['total']} photos to {self.output_dir}…")
        
        for idx, photo in enumerate(photos, 1):
            if idx % batch_size == 0 or idx == 1:
                logger.info(f"Progress: {idx}/{stats['total']}")
            
            mime_type = photo.get('mimeType', '')
            if not mime_type.startswith('image/') and not mime_type.startswith('video/'):
                logger.debug(f"Skipping unknown media type: {mime_type}")
                stats['skipped'] += 1
                continue
            
            result = self.download_photo(photo, preserve_metadata)
            if result:
                file_path, _ = result
                stats['successful'] += 1
                stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                stats['downloaded_files'].append(str(file_path))
            else:
                stats['failed'] += 1
        
        stats['end_time'] = datetime.now()
        stats['duration_seconds'] = (stats['end_time'] - stats['start_time']).total_seconds()
        
        logger.info("=" * 60)
        logger.info("Download Complete!")
        logger.info(f"  Downloaded: {stats['successful']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        logger.info(f"  Total size: {stats['total_size_mb']:.1f}MB")
        logger.info(f"  Duration: {stats['duration_seconds']:.1f}s")
        logger.info("=" * 60)
        
        return stats
