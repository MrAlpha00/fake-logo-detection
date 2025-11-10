"""
Authentication and authorization module for Fake Logo Detection Suite.
Provides user management, role-based access control, and session handling.
"""
import hashlib
import sqlite3
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from src.utils import get_logger

logger = get_logger(__name__)

# Try to import argon2, fall back to PBKDF2 if not available
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHash
    HAS_ARGON2 = True
except ImportError:
    HAS_ARGON2 = False
    logger.warning("argon2-cffi not installed, falling back to PBKDF2")


class UserRole:
    """User role definitions with permissions."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    
    @classmethod
    def get_all_roles(cls):
        """Return list of all available roles."""
        return [cls.ADMIN, cls.ANALYST, cls.VIEWER]
    
    @classmethod
    def get_permissions(cls, role):
        """
        Get permissions for a given role.
        
        Permissions:
        - view_detections: View detection results and analytics
        - run_analysis: Run logo detection and analysis
        - export_data: Export CSV/Excel reports
        - manage_users: Create, edit, delete users
        - configure_settings: Change system settings
        """
        permissions = {
            cls.ADMIN: {
                'view_detections': True,
                'run_analysis': True,
                'export_data': True,
                'manage_users': True,
                'configure_settings': True,
                'delete_records': True,
            },
            cls.ANALYST: {
                'view_detections': True,
                'run_analysis': True,
                'export_data': True,
                'manage_users': False,
                'configure_settings': False,
                'delete_records': False,
            },
            cls.VIEWER: {
                'view_detections': True,
                'run_analysis': False,
                'export_data': False,
                'manage_users': False,
                'configure_settings': False,
                'delete_records': False,
            }
        }
        return permissions.get(role, {})


class AuthManager:
    """
    Manages user authentication and authorization.
    """
    
    def __init__(self, db_path='users.db'):
        """
        Initialize authentication manager.
        
        Args:
            db_path: Path to SQLite database for user storage
        """
        self.db_path = db_path
        self._init_database()
        self._create_default_admin()
    
    def _init_database(self):
        """Initialize user database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
            logger.info("User database initialized")
    
    def _create_default_admin(self):
        """Create default admin user if none exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = ?", (UserRole.ADMIN,))
            admin_count = cursor.fetchone()[0]
            
            if admin_count == 0:
                # Create default admin: username=admin, password=admin (should be changed!)
                self.create_user('admin', 'admin@example.com', 'admin', UserRole.ADMIN)
                logger.warning("Default admin user created (username: admin, password: admin)")
                logger.warning("IMPORTANT: Change the default admin password immediately!")
    
    def _hash_password(self, password):
        """
        Hash password using Argon2 (preferred) or PBKDF2 with HMAC-SHA256 (fallback).
        
        Args:
            password: Plain text password
        
        Returns:
            str: Hashed password
        """
        if HAS_ARGON2:
            # Use Argon2id (best practice for password hashing)
            ph = PasswordHasher()
            return ph.hash(password)
        else:
            # Fallback to PBKDF2-HMAC-SHA256 with 100,000 iterations
            salt = secrets.token_bytes(32)
            iterations = 100000
            key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations)
            # Store as pbkdf2:iterations:salt:hash
            return f"pbkdf2:{iterations}:{salt.hex()}:{key.hex()}"
    
    def _verify_password(self, password, stored_hash):
        """
        Verify password against stored hash.
        Supports both Argon2 and PBKDF2 formats.
        
        Args:
            password: Plain text password to verify
            stored_hash: Stored hash (Argon2 or PBKDF2 format)
        
        Returns:
            bool: True if password matches
        """
        try:
            # Check if it's Argon2 format
            if stored_hash.startswith('$argon2'):
                if not HAS_ARGON2:
                    logger.error("Argon2 hash found but argon2-cffi not installed")
                    return False
                
                ph = PasswordHasher()
                try:
                    ph.verify(stored_hash, password)
                    # Optionally rehash if parameters changed
                    if ph.check_needs_rehash(stored_hash):
                        logger.info("Password needs rehashing with updated parameters")
                    return True
                except (VerifyMismatchError, VerificationError, InvalidHash):
                    return False
            
            # Check if it's PBKDF2 format
            elif stored_hash.startswith('pbkdf2:'):
                parts = stored_hash.split(':')
                if len(parts) != 4:
                    logger.error("Invalid PBKDF2 hash format")
                    return False
                
                _, iterations_str, salt_hex, stored_key_hex = parts
                iterations = int(iterations_str)
                salt = bytes.fromhex(salt_hex)
                stored_key = bytes.fromhex(stored_key_hex)
                
                # Compute key with same parameters
                computed_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations)
                
                # Constant-time comparison
                return secrets.compare_digest(computed_key, stored_key)
            
            # Legacy SHA256 format (for backward compatibility)
            else:
                logger.warning("Using legacy SHA256 hash verification - please rehash passwords")
                salt, hashed = stored_hash.split(':', 1)
                password_with_salt = f"{password}{salt}"
                hash_obj = hashlib.sha256(password_with_salt.encode())
                computed_hash = hash_obj.hexdigest()
                return computed_hash == hashed
                
        except (ValueError, TypeError) as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    def create_user(self, username, email, password, role=UserRole.VIEWER):
        """
        Create a new user.
        
        Args:
            username: Unique username
            email: User email
            password: Plain text password (will be hashed)
            role: User role (default: viewer)
        
        Returns:
            bool: True if user created successfully
        """
        if role not in UserRole.get_all_roles():
            logger.error(f"Invalid role: {role}")
            return False
        
        password_hash = self._hash_password(password)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES (?, ?, ?, ?)
                ''', (username, email, password_hash, role))
                conn.commit()
                logger.info(f"User created: {username} ({role})")
                return True
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to create user: {e}")
            return False
    
    def authenticate(self, username, password):
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Plain text password
        
        Returns:
            dict: User information if authenticated, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, email, password_hash, role, is_active
                FROM users
                WHERE username = ?
            ''', (username,))
            
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Authentication failed: user not found ({username})")
                return None
            
            user_id, username, email, password_hash, role, is_active = result
            
            if not is_active:
                logger.warning(f"Authentication failed: user inactive ({username})")
                return None
            
            if not self._verify_password(password, password_hash):
                logger.warning(f"Authentication failed: wrong password ({username})")
                return None
            
            # Update last login
            cursor.execute('''
                UPDATE users
                SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (user_id,))
            conn.commit()
            
            logger.info(f"User authenticated: {username}")
            
            return {
                'id': user_id,
                'username': username,
                'email': email,
                'role': role,
                'permissions': UserRole.get_permissions(role)
            }
    
    def create_session(self, user_id, duration_hours=24):
        """
        Create a session token for user.
        
        Args:
            user_id: User ID
            duration_hours: Session duration in hours
        
        Returns:
            str: Session token
        """
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=duration_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at))
            conn.commit()
        
        logger.info(f"Session created for user {user_id}")
        return session_token
    
    def validate_session(self, session_token):
        """
        Validate session token and return user info.
        
        Args:
            session_token: Session token to validate
        
        Returns:
            dict: User information if valid, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.user_id, u.username, u.email, u.role, s.expires_at
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ? AND u.is_active = 1
            ''', (session_token,))
            
            result = cursor.fetchone()
            
            if not result:
                return None
            
            user_id, username, email, role, expires_at = result
            
            # Check if session expired
            expires_dt = datetime.fromisoformat(expires_at)
            if datetime.now() > expires_dt:
                logger.warning(f"Session expired for user {username}")
                return None
            
            return {
                'id': user_id,
                'username': username,
                'email': email,
                'role': role,
                'permissions': UserRole.get_permissions(role)
            }
    
    def logout(self, session_token):
        """
        Logout user by removing session.
        
        Args:
            session_token: Session token to remove
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sessions WHERE session_token = ?', (session_token,))
            conn.commit()
        
        logger.info("User logged out")
    
    def get_all_users(self):
        """
        Get all users (admin only).
        
        Returns:
            list: List of user dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, email, role, created_at, last_login, is_active
                FROM users
                ORDER BY created_at DESC
            ''')
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    'id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'role': row[3],
                    'created_at': row[4],
                    'last_login': row[5],
                    'is_active': bool(row[6])
                })
            
            return users
    
    def update_user_role(self, user_id, new_role):
        """
        Update user role (admin only).
        
        Args:
            user_id: User ID to update
            new_role: New role
        
        Returns:
            bool: True if updated successfully
        """
        if new_role not in UserRole.get_all_roles():
            return False
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users
                SET role = ?
                WHERE id = ?
            ''', (new_role, user_id))
            conn.commit()
        
        logger.info(f"User {user_id} role updated to {new_role}")
        return True
    
    def deactivate_user(self, user_id):
        """
        Deactivate user account (admin only).
        
        Args:
            user_id: User ID to deactivate
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET is_active = 0 WHERE id = ?', (user_id,))
            cursor.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
            conn.commit()
        
        logger.info(f"User {user_id} deactivated")
    
    def change_password(self, user_id, old_password, new_password):
        """
        Change user password.
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
        
        Returns:
            bool: True if password changed successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,))
            result = cursor.fetchone()
            
            if not result:
                return False
            
            password_hash = result[0]
            
            # Verify old password
            if not self._verify_password(old_password, password_hash):
                logger.warning(f"Password change failed: wrong old password (user {user_id})")
                return False
            
            # Update to new password
            new_hash = self._hash_password(new_password)
            cursor.execute('''
                UPDATE users
                SET password_hash = ?
                WHERE id = ?
            ''', (new_hash, user_id))
            conn.commit()
        
        logger.info(f"Password changed for user {user_id}")
        return True
