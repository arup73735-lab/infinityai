"""
Authentication and authorization using OAuth2 + JWT.

Implements secure token-based authentication with:
- JWT token generation and validation
- Password hashing with bcrypt
- Token expiration and refresh
- User management (simplified for demo)
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: list[str] = []


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    is_admin: bool = False


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


# Demo users database (replace with real database in production)
fake_users_db = {
    "demo": {
        "username": "demo",
        "full_name": "Demo User",
        "email": "demo@example.com",
        "hashed_password": pwd_context.hash("demo123"),
        "disabled": False,
        "is_admin": False,
    },
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin123"),
        "disabled": False,
        "is_admin": True,
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        scopes: list = payload.get("scopes", [])
        
        if username is None:
            return None
            
        return TokenData(
            username=username,
            user_id=user_id,
            scopes=scopes
        )
    except JWTError:
        return None


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[User]:
    """Get current user from token."""
    if token is None:
        return None
    
    token_data = decode_token(token)
    if token_data is None or token_data.username is None:
        return None
    
    user = get_user(token_data.username)
    if user is None:
        return None
    
    return User(**user.dict())


async def get_current_active_user(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """Get current active user (required)."""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current admin user (required)."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return current_user


def create_user(username: str, password: str, **kwargs) -> UserInDB:
    """Create a new user."""
    if username in fake_users_db:
        raise ValueError("User already exists")
    
    user_dict = {
        "username": username,
        "hashed_password": get_password_hash(password),
        "disabled": False,
        "is_admin": False,
        **kwargs
    }
    
    fake_users_db[username] = user_dict
    return UserInDB(**user_dict)
