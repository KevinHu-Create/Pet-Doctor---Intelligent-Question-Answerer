from app.db.database import engine
from app.db.model import Base

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("tables created")