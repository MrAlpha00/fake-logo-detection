# Authentication Integration Guide

This guide explains the authentication and authorization system for the Fake Logo Detection Suite.

## Overview

The authentication system provides:
- **User Management**: Create, deactivate, and manage users
- **Role-Based Access Control (RBAC)**: Three roles with different permission levels
- **Session Management**: Secure session handling with expiration
- **Password Security**: SHA256 hashing with salts

## User Roles

### Viewer
- **Permissions**: View detection results and analytics only
- **Cannot**: Run new analyses, export data, or manage users
- **Use Case**: Stakeholders who need read-only access to results

### Analyst
- **Permissions**: Run analyses, view results, export data
- **Cannot**: Manage users or system settings
- **Use Case**: Forensic analysts and investigators

### Admin
- **Permissions**: Full access to all features
- **Can**: Manage users, configure settings, delete records
- **Use Case**: System administrators

## Quick Start

### Default Credentials

```
Username: admin
Password: admin
```

⚠️ **IMPORTANT**: Change the default password immediately after first login!

### Creating Users

Admins can create new users through the User Management tab:

1. Navigate to User Management (admin only)
2. Click "Create New User"
3. Fill in username, email, password, and role
4. Submit

Users can also self-register with Viewer role by default.

## Architecture

### Components

1. **`src/auth.py`** - Core authentication logic
   - `AuthManager`: Main authentication class
   - `UserRole`: Role definitions and permissions
   - Password hashing and verification
   - Session token management

2. **`src/auth_streamlit.py`** - Streamlit UI components
   - Login/signup pages
   - User sidebar
   - Permission checks
   - User management interface

3. **`users.db`** - SQLite database
   - `users` table: User accounts
   - `sessions` table: Active sessions

### Database Schema

**Users Table:**
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN
)
```

**Sessions Table:**
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    session_token TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
```

## Integration with Streamlit App

### Basic Integration Pattern

```python
from src.auth import AuthManager
from src.auth_streamlit import (
    init_session_state, show_login_page, show_user_sidebar,
    check_permission, require_permission
)

def main():
    # Initialize
    auth_manager = AuthManager()
    init_session_state()
    
    # Show login if not authenticated
    if not st.session_state.authenticated:
        show_login_page(auth_manager)
        return
    
    # Show user sidebar
    show_user_sidebar(auth_manager)
    
    # Main app content with permission checks
    if require_permission('run_analysis'):
        # Show analysis features
        pass
    
    if require_permission('export_data'):
        # Show export features
        pass
```

### Permission-Based UI

```python
# Hide features based on permissions
if check_permission('run_analysis'):
    st.button("Analyze Image")
else:
    st.info("🔒 Analysis is restricted to Analyst role and above")

# Require permission before allowing action
if st.button("Export Data"):
    if not require_permission('export_data', "Only Analysts and Admins can export data"):
        st.stop()
    
    # Proceed with export
    ...
```

## Permission Matrix

| Feature | Viewer | Analyst | Admin |
|---------|--------|---------|-------|
| View Detections | ✅ | ✅ | ✅ |
| Run Analysis | ❌ | ✅ | ✅ |
| Export Data | ❌ | ✅ | ✅ |
| Manage Users | ❌ | ❌ | ✅ |
| Configure Settings | ❌ | ❌ | ✅ |
| Delete Records | ❌ | ❌ | ✅ |

## Security Features

### Password Security
- SHA256 hashing with random salts
- Salt stored with hash in format `salt:hash`
- Passwords never stored in plain text

### Session Security
- Cryptographically secure session tokens (URL-safe, 32 bytes)
- 24-hour session expiration by default
- Sessions automatically cleaned up on logout

### Account Security
- Account deactivation (vs deletion) for audit trail
- Prevent self-deactivation for admins
- Last login tracking

## API Reference

### AuthManager

```python
auth = AuthManager(db_path='users.db')

# Authentication
user = auth.authenticate(username, password)
session_token = auth.create_session(user['id'], duration_hours=24)
user = auth.validate_session(session_token)
auth.logout(session_token)

# User Management
auth.create_user(username, email, password, role=UserRole.VIEWER)
users = auth.get_all_users()
auth.update_user_role(user_id, new_role)
auth.deactivate_user(user_id)
auth.change_password(user_id, old_password, new_password)
```

### Streamlit Helpers

```python
# Session State
init_session_state()

# UI Components
show_login_page(auth_manager)
show_signup_form(auth_manager)
show_user_sidebar(auth_manager)
show_user_management_tab(auth_manager)
show_password_change_form(auth_manager)

# Permission Checks
has_permission = check_permission('permission_name')
if require_permission('permission_name', "Custom error message"):
    # Proceed with protected action
    pass
```

## Deployment Considerations

### Production Setup

1. **Change Default Credentials**
   ```python
   auth.change_password(admin_user_id, 'admin', 'strong_password_here')
   ```

2. **Backup Users Database**
   - Regularly backup `users.db`
   - Store backups securely (encrypted)

3. **Session Timeout**
   - Adjust session duration based on security requirements
   - Shorter sessions = more security, less convenience

4. **SSL/TLS**
   - Always use HTTPS in production
   - Passwords transmitted over encrypted connection

### Multi-Environment Setup

**Development:**
```python
auth = AuthManager(db_path='users_dev.db')
```

**Production:**
```python
auth = AuthManager(db_path='/secure/path/users_prod.db')
```

## Troubleshooting

### Common Issues

**Cannot login with default credentials:**
- Ensure `users.db` exists and contains default admin
- Check database permissions
- Delete `users.db` to regenerate default admin

**Session expires immediately:**
- Check system clock synchronization
- Verify session duration setting
- Check for timezone issues

**Password hash format error:**
- Ensure password_hash contains colon separator
- Regenerate user if hash is corrupted

**Permissions not working:**
- Verify user role is correctly set
- Check UserRole.get_permissions() mapping
- Ensure session is valid and not expired

## Future Enhancements

Potential improvements for production systems:

1. **OAuth Integration**: Google, GitHub, Microsoft authentication
2. **Two-Factor Authentication**: TOTP or SMS-based 2FA
3. **Audit Logging**: Track all user actions
4. **Password Policies**: Enforce complexity, expiration
5. **Account Recovery**: Email-based password reset
6. **API Key Authentication**: For programmatic access
7. **IP Whitelisting**: Restrict access by IP range
8. **Session Management UI**: View/revoke active sessions

## License & Compliance

When implementing authentication:
- Comply with data protection regulations (GDPR, CCPA)
- Inform users about data collection
- Provide data export/deletion capabilities
- Implement proper access controls for PII
- Regular security audits and updates
