from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
try:
    from database import Base
except ImportError:
    from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=True)
    name = Column(String, nullable=True)
    profile_pic = Column(String, nullable=True)
    google_id = Column(String, unique=True, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversations = relationship("Conversation", back_populates="user")
    memories = relationship("Memory", back_populates="user")
    logs = relationship("Log", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    title = Column(String)
    chat_history = Column(Text, default="[]") # Store full history as JSON STRING
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    user = relationship("User", back_populates="conversations")

class Memory(Base):
    __tablename__ = "memory"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    content = Column(Text)
    vector = Column(Text) # JSON string of embeddings [0.1, 0.2, ...]
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="memories")

class Log(Base):
    __tablename__ = "logs"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    activity = Column(String) # e.g., "login", "chat", "upload"
    data = Column(Text, nullable=True) # JSON details
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="logs")

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    file_path = Column(String)
    extracted_text = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="processed")
