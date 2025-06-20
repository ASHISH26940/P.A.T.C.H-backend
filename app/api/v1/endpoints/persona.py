# app/api/v1/endpoints/persona.py
from fastapi import APIRouter, HTTPException, status, Depends
from loguru import logger
from typing import List
import uuid
from sqlalchemy import select

# Import your Pydantic schemas
from app.schemas.persona import PersonaCreate, PersonaInDB, PersonaUpdate

# Import your SQLAlchemy ORM model
from app.models.persona import Persona # Assuming your SQLAlchemy model is named Persona

# Import the database session dependency
from app.core.database import get_db, AsyncSession # Import AsyncSession too

router = APIRouter()

# --- NO MORE IN-MEMORY DICTIONARY! ---
# Remove the _personas_db dictionary, as we will now interact with the database.
# -------------------------------------

@router.post(
    "/",
    response_model=PersonaInDB,
    status_code=status.HTTP_201_CREATED,
    summary="Create Persona"
)
async def create_persona(
    persona_data: PersonaCreate,
    db: AsyncSession = Depends(get_db) # Inject the database session
):
    """
    Creates a new persona and saves it to the PostgreSQL database.
    """
    # Create an instance of your SQLAlchemy Persona model
    # Convert Pydantic model to a dictionary to pass to SQLAlchemy model constructor
    db_persona = Persona(**persona_data.model_dump())
    
    db.add(db_persona) # Add the new persona to the session
    await db.commit() # Commit the transaction to save to DB
    await db.refresh(db_persona) # Refresh the object to get its generated ID from the DB
    
    logger.info(f"Created persona: {db_persona.name} with ID: {db_persona.id} in the database.")
    
    return PersonaInDB.model_validate(db_persona) # Convert SQLAlchemy model to Pydantic for response

@router.get(
    "/{persona_id}",
    response_model=PersonaInDB,
    summary="Retrieve Persona by ID"
)
async def get_persona(
    persona_id: uuid.UUID,
    db: AsyncSession = Depends(get_db) # Inject the database session
):
    """
    Retrieves a persona by its ID from the PostgreSQL database.
    """
    # Query the database for the persona by ID
    db_persona = await db.get(Persona, persona_id) # AsyncSession.get() is a direct fetch by PK
    
    if not db_persona:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Persona not found")
        
    return PersonaInDB.model_validate(db_persona) # Convert SQLAlchemy model to Pydantic for response

@router.get(
    "/",
    response_model=List[PersonaInDB],
    summary="Retrieve All Personas"
)
async def get_all_personas(
    db: AsyncSession = Depends(get_db) # Inject the database session
):
    """
    Retrieves a list of all personas from the PostgreSQL database.
    """
    # Build a simple select statement
    result = await db.execute(select(Persona))
    personas = result.scalars().all() # Get all Persona objects
    
    # Convert list of SQLAlchemy models to list of Pydantic models
    return [PersonaInDB.model_validate(p) for p in personas]

@router.put(
    "/{persona_id}",
    response_model=PersonaInDB,
    summary="Update Persona by ID"
)
async def update_persona(
    persona_id: uuid.UUID,
    persona_update: PersonaUpdate,
    db: AsyncSession = Depends(get_db) # Inject the database session
):
    """
    Updates an existing persona in the PostgreSQL database.
    """
    db_persona = await db.get(Persona, persona_id)
    if not db_persona:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Persona not found")
    
    # Update fields from persona_update Pydantic model to SQLAlchemy ORM model
    update_data = persona_update.model_dump(exclude_unset=True) # Only update fields that are provided
    for key, value in update_data.items():
        setattr(db_persona, key, value)
    
    db.add(db_persona) # Re-add to session to mark as dirty (optional but good practice)
    await db.commit() # Commit changes to DB
    await db.refresh(db_persona) # Refresh to get latest state from DB
    
    logger.info(f"Updated persona with ID: {persona_id} in the database.")
    return PersonaInDB.model_validate(db_persona)

@router.delete(
    "/{persona_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Persona by ID"
)
async def delete_persona(
    persona_id: uuid.UUID,
    db: AsyncSession = Depends(get_db) # Inject the database session
):
    """
    Deletes a persona from the PostgreSQL database.
    """
    db_persona = await db.get(Persona, persona_id)
    if not db_persona:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Persona not found for deletion")
    
    await db.delete(db_persona) # Mark for deletion
    await db.commit() # Commit deletion to DB
    
    logger.info(f"Deleted persona with ID: {persona_id} from the database.")
    return # 204 No Content response