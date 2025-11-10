"""
Streamlit authentication UI components and session management.
Provides login/signup pages and role-based access control for the main app.
"""
import streamlit as st
from src.auth import AuthManager, UserRole


def init_session_state():
    """Initialize session state variables for authentication."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'session_token' not in st.session_state:
        st.session_state.session_token = None


def check_permission(permission_name):
    """
    Check if current user has a specific permission.
    
    Args:
        permission_name: Name of permission to check
    
    Returns:
        bool: True if user has permission
    """
    if not st.session_state.authenticated or not st.session_state.user:
        return False
    
    return st.session_state.user['permissions'].get(permission_name, False)


def require_permission(permission_name, message="You don't have permission to access this feature."):
    """
    Decorator/helper to require a specific permission.
    Shows warning if user doesn't have permission.
    
    Args:
        permission_name: Permission required
        message: Message to show if permission denied
    
    Returns:
        bool: True if user has permission, False otherwise
    """
    if not check_permission(permission_name):
        st.warning(f"🔒 {message}")
        return False
    return True


def show_login_page(auth_manager):
    """
    Display login page.
    
    Args:
        auth_manager: AuthManager instance
    """
    st.markdown('<div class="main-header">🔍 Fake Logo Detection Suite</div>', 
                unsafe_allow_html=True)
    st.markdown("### 🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
                return
            
            user = auth_manager.authenticate(username, password)
            
            if user:
                # Create session
                session_token = auth_manager.create_session(user['id'])
                
                # Store in session state
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.session_token = session_token
                
                st.success(f"Welcome, {user['username']}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    st.markdown("---")
    
    # Signup section
    with st.expander("Don't have an account? Sign up"):
        show_signup_form(auth_manager)
    
    # Default credentials reminder
    st.info("""
    **Default Admin Credentials:**  
    Username: `admin`  
    Password: `admin`
    
    ⚠️ **Important:** Change the default password after first login!
    """)


def show_signup_form(auth_manager):
    """
    Display signup form.
    
    Args:
        auth_manager: AuthManager instance
    """
    st.markdown("### 📝 Create Account")
    
    with st.form("signup_form"):
        new_username = st.text_input("Username", key="signup_username")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        signup_submit = st.form_submit_button("Create Account", use_container_width=True)
        
        if signup_submit:
            if not all([new_username, new_email, new_password, confirm_password]):
                st.error("Please fill in all fields")
                return
            
            if new_password != confirm_password:
                st.error("Passwords do not match")
                return
            
            if len(new_password) < 6:
                st.error("Password must be at least 6 characters")
                return
            
            # Create user with Viewer role by default
            success = auth_manager.create_user(
                new_username, new_email, new_password, UserRole.VIEWER
            )
            
            if success:
                st.success("Account created successfully! Please log in.")
                st.balloons()
            else:
                st.error("Username or email already exists")


def show_user_sidebar(auth_manager):
    """
    Display user info and logout button in sidebar.
    
    Args:
        auth_manager: AuthManager instance
    """
    if st.session_state.authenticated and st.session_state.user:
        user = st.session_state.user
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 Current User")
        st.sidebar.write(f"**{user['username']}**")
        st.sidebar.caption(f"Role: {user['role'].title()}")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            # Logout
            if st.session_state.session_token:
                auth_manager.logout(st.session_state.session_token)
            
            # Clear session
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.session_token = None
            
            st.rerun()


def show_user_management_tab(auth_manager):
    """
    Display user management interface (admin only).
    
    Args:
        auth_manager: AuthManager instance
    """
    if not check_permission('manage_users'):
        st.warning("🔒 This section is only accessible to administrators.")
        return
    
    st.subheader("👥 User Management")
    
    # Get all users
    users = auth_manager.get_all_users()
    
    if not users:
        st.info("No users found")
        return
    
    st.markdown(f"**Total Users:** {len(users)}")
    st.markdown("---")
    
    # Display users in table
    for user in users:
        with st.expander(f"{user['username']} ({user['email']}) - {user['role'].title()}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**ID:** {user['id']}")
                st.write(f"**Created:** {user['created_at']}")
            
            with col2:
                st.write(f"**Role:** {user['role']}")
                st.write(f"**Active:** {'Yes' if user['is_active'] else 'No'}")
            
            with col3:
                st.write(f"**Last Login:** {user['last_login'] or 'Never'}")
            
            # Admin actions
            st.markdown("---")
            st.markdown("**Admin Actions:**")
            
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                # Change role
                new_role = st.selectbox(
                    "Change Role",
                    UserRole.get_all_roles(),
                    index=UserRole.get_all_roles().index(user['role']),
                    key=f"role_{user['id']}"
                )
                
                if st.button("Update Role", key=f"update_role_{user['id']}"):
                    if auth_manager.update_user_role(user['id'], new_role):
                        st.success(f"Role updated to {new_role}")
                        st.rerun()
                    else:
                        st.error("Failed to update role")
            
            with action_col2:
                # Deactivate user
                if user['is_active']:
                    if st.button("Deactivate User", key=f"deactivate_{user['id']}", type="secondary"):
                        # Prevent self-deactivation
                        if user['id'] == st.session_state.user['id']:
                            st.error("You cannot deactivate your own account!")
                        else:
                            auth_manager.deactivate_user(user['id'])
                            st.success(f"User {user['username']} deactivated")
                            st.rerun()
                else:
                    st.caption("User is deactivated")
    
    st.markdown("---")
    
    # Create new user
    with st.expander("➕ Create New User"):
        with st.form("create_user_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", UserRole.get_all_roles())
            
            if st.form_submit_button("Create User"):
                if auth_manager.create_user(username, email, password, role):
                    st.success(f"User {username} created successfully!")
                    st.rerun()
                else:
                    st.error("Failed to create user (username or email may already exist)")


def show_password_change_form(auth_manager):
    """
    Display password change form.
    
    Args:
        auth_manager: AuthManager instance
    """
    st.subheader("🔑 Change Password")
    
    with st.form("change_password_form"):
        old_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_new = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Change Password"):
            if not all([old_password, new_password, confirm_new]):
                st.error("Please fill in all fields")
                return
            
            if new_password != confirm_new:
                st.error("New passwords do not match")
                return
            
            if len(new_password) < 6:
                st.error("Password must be at least 6 characters")
                return
            
            user_id = st.session_state.user['id']
            
            if auth_manager.change_password(user_id, old_password, new_password):
                st.success("Password changed successfully!")
            else:
                st.error("Failed to change password. Please check your current password.")
