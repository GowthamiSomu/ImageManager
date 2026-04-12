"""
Script to download all photos from Google Photos and store them in D:\Photos folder.
This script handles OAuth2 authentication and batch downloads.

Usage:
    # Using credentials file
    python scripts/download_from_google_photos.py --creds-file google_credentials.json
    
    # Using client ID and secret directly
    python scripts/download_from_google_photos.py --client-id YOUR_ID --client-secret YOUR_SECRET
    
    # Using environment variables
    set GOOGLE_CLIENT_ID=YOUR_ID
    set GOOGLE_CLIENT_SECRET=YOUR_SECRET
    python scripts/download_from_google_photos.py

Examples:
    python scripts/download_from_google_photos.py --max-photos 100
    python scripts/download_from_google_photos.py --output-dir "D:\\Photos"
    python scripts/download_from_google_photos.py --client-id xxx --client-secret yyy --max-photos 50
"""
import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.google_photos_service import GooglePhotosService
from infrastructure.logging import setup_logging
from infrastructure.config import Config


def setup_google_credentials() -> dict:
    """
    Check for Google credentials from file, environment, or command-line args.
    
    Returns:
        Dictionary with 'credentials_file', 'client_id', and 'client_secret' keys
    """
    credentials = {
        'credentials_file': None,
        'client_id': None,
        'client_secret': None
    }
    
    # Check environment variables
    credentials['client_id'] = os.environ.get('GOOGLE_CLIENT_ID')
    credentials['client_secret'] = os.environ.get('GOOGLE_CLIENT_SECRET')
    
    # Check for credentials file
    if Path('google_credentials.json').exists():
        credentials['credentials_file'] = 'google_credentials.json'
    
    return credentials


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download all photos from Google Photos to D:\\Photos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using credentials file
    python scripts/download_from_google_photos.py --creds-file google_credentials.json
    
    # Using client ID and secret
    python scripts/download_from_google_photos.py --client-id YOUR_ID --client-secret YOUR_SECRET
    
    # Using environment variables
    set GOOGLE_CLIENT_ID=YOUR_ID
    set GOOGLE_CLIENT_SECRET=YOUR_SECRET
    python scripts/download_from_google_photos.py
    
    # With filters
    python scripts/download_from_google_photos.py --client-id xxx --client-secret yyy --max-photos 100
        """
    )
    
    parser.add_argument(
        '--max-photos',
        type=int,
        default=None,
        help='Maximum number of photos to download (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='D:\\Photos',
        help='Output directory for downloaded photos (default: D:\\Photos)'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Do not save metadata JSON files (default: save metadata)'
    )
    parser.add_argument(
        '--creds-file',
        type=str,
        default=None,
        help='Path to Google OAuth2 credentials JSON file'
    )
    parser.add_argument(
        '--client-id',
        type=str,
        default=None,
        help='Google OAuth2 Client ID (use instead of credentials file)'
    )
    parser.add_argument(
        '--client-secret',
        type=str,
        default=None,
        help='Google OAuth2 Client Secret (use instead of credentials file)'
    )
    parser.add_argument(
        '--order-by',
        type=str,
        default='creationTime',
        choices=['creationTime', 'relevance'],
        help='Sort order for photos (default: creationTime for oldest first)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level='INFO',
        log_file=Path('logs') / f"google_photos_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 70)
        logger.info("Google Photos Download Script")
        logger.info("=" * 70)
        
        # Determine credentials to use
        creds_file = args.creds_file
        client_id = args.client_id
        client_secret = args.client_secret
        
        # Check environment variables if not provided as arguments
        if not client_id:
            client_id = os.environ.get('GOOGLE_CLIENT_ID')
        if not client_secret:
            client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
        
        # Verify we have credentials
        if not creds_file and not (client_id and client_secret):
            print("\n" + "=" * 70)
            print("ERROR: No credentials provided!")
            print("=" * 70)
            print("""
Please provide credentials using one of these methods:

1. Credentials File (downloaded from Google Cloud Console):
   python scripts/download_from_google_photos.py --creds-file google_credentials.json

2. Client ID and Secret (from Google Cloud Console):
   python scripts/download_from_google_photos.py --client-id YOUR_ID --client-secret YOUR_SECRET

3. Environment Variables:
   set GOOGLE_CLIENT_ID=YOUR_ID
   set GOOGLE_CLIENT_SECRET=YOUR_SECRET
   python scripts/download_from_google_photos.py

For setup instructions, see: GOOGLE_PHOTOS_SETUP.md
            """)
            print("=" * 70 + "\n")
            sys.exit(1)
        
        logger.info("Initializing Google Photos service...")
        logger.info(f"  Credentials: {'credentials_file' if creds_file else 'client_id/secret'}")
        logger.info(f"  Output directory: {args.output_dir}")
        logger.info(f"  Max photos: {args.max_photos or 'unlimited'}")
        logger.info(f"  Save metadata: {not args.no_metadata}")
        
        # Initialize service
        service = GooglePhotosService(
            credentials_file=creds_file,
            client_id=client_id,
            client_secret=client_secret,
            token_file='.google_photos_token.json',
            output_dir=args.output_dir
        )
        
        # Download photos
        logger.info(f"Starting download (order_by={args.order_by})...")
        stats = service.download_all_photos(
            max_photos=args.max_photos,
            preserve_metadata=not args.no_metadata,
            order_by=args.order_by
        )
        
        # Return exit code based on results
        if stats['failed'] == 0:
            logger.info("All photos downloaded successfully!")
            return 0
        else:
            logger.warning(f"Download completed with {stats['failed']} failures")
            return 1
            
    except FileNotFoundError as e:
        logger.error(f"Setup error: {e}")
        print(f"\nERROR: {e}")
        return 1
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
