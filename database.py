from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./health_data.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Nigerian time is UTC+1 (West Africa Time - WAT)
def get_nigerian_time():
    nigerian_tz = datetime.timezone(datetime.timedelta(hours=1))
    return datetime.datetime.now(nigerian_tz)

class HealthReading(Base):
    __tablename__ = "readings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, default="user_001")
    heart_rate = Column(Float)
    spo2 = Column(Float)
    steps = Column(Integer)
    sleep_hours = Column(Float)
    activity_level = Column(String)
    risk_level = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=get_nigerian_time)

class UserBaseline(Base):
    __tablename__ = "baselines"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True)
    avg_heart_rate = Column(Float)
    avg_spo2 = Column(Float)
    avg_steps = Column(Float)
    avg_sleep = Column(Float)
    hr_std = Column(Float)
    spo2_std = Column(Float)
    created_at = Column(DateTime, default=get_nigerian_time)
    updated_at = Column(DateTime, default=get_nigerian_time)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()