from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .face_recognizer_arcface import FaceEmbedding, FaceRecognizer

@dataclass
class PersonEntry:
    person_id: str                  # Logic ID in the system
    name: Optional[str] = None
    age: Optional[int] = None
    metadata: Dict = field(default_factory=dict)
    embeddings: List[FaceEmbedding] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        """Average vector of embeddings for fast matching."""
        if not self.embeddings:
            return np.zeros(0, dtype=np.float32)
        vectors = np.stack([e.vector for e in self.embeddings], axis=0)
        return vectors.mean(axis=0)

class FaceRegistry:
    """
    Manages face embedding database:
    - Register new persons.
    - Update existing embeddings.
    - Identify nearest person for input face.
    """

    def __init__(self, 
                 recognizer: FaceRecognizer,
                 similarity_threshold: float = 0.5):
        self.recognizer = recognizer
        self.similarity_threshold = similarity_threshold
        self._persons: Dict[str, PersonEntry] = {}

    def register_person(
        self,
        person_id: str,
        face_img,
        name: Optional[str] = None,
        age: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> PersonEntry:
        emb = self.recognizer.embed_face(face_img)

        if person_id not in self._persons:
            self._persons[person_id] = PersonEntry(
                person_id=person_id,
                name=name,
                age=age,
                metadata=metadata or {},
                embeddings=[emb],
            )
        else:
            entry = self._persons[person_id]
            entry.embeddings.append(emb)
            if name is not None:
                entry.name = name
            if age is not None:
                entry.age = age
            if metadata:
                entry.metadata.update(metadata)

        return self._persons[person_id]

    def remove_person(self, person_id: str) -> bool:
        return self._persons.pop(person_id, None) is not None

    def get_person(self, person_id: str) -> Optional[PersonEntry]:
        return self._persons.get(person_id)

    def list_persons(self) -> List[PersonEntry]:
        return list(self._persons.values())

    def identify(self, face_img, top_k: int = 1) -> List[Tuple[PersonEntry, float]]:
        emb = self.recognizer.embed_face(face_img)
        return self.identify_from_embedding(emb, top_k=top_k)

    def identify_from_embedding(self, emb: FaceEmbedding, top_k: int = 1) -> List[Tuple[PersonEntry, float]]:
        scores: List[Tuple[PersonEntry, float]] = []
        for entry in self._persons.values():
            if entry.centroid.size == 0:
                continue
            sim = float(
                np.dot(emb.vector, entry.centroid)
                / (np.linalg.norm(emb.vector) * np.linalg.norm(entry.centroid))
            )
            scores.append((entry, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        if top_k > 0:
            scores = scores[:top_k]
        return scores

    def is_match(self, emb: FaceEmbedding, person_id: str) -> bool:
        entry = self._persons.get(person_id)
        if not entry or entry.centroid.size == 0:
            return False
        sim = float(
            np.dot(emb.vector, entry.centroid)
            / (np.linalg.norm(emb.vector) * np.linalg.norm(entry.centroid))
        )
        return sim >= self.similarity_threshold

    def save(self, path: str) -> None:
        # TODO: Implement serialization
        pass

    @classmethod
    def load(cls, path: str, recognizer: FaceRecognizer, similarity_threshold: float = 0.5):
        registry = cls(recognizer, similarity_threshold)
        # TODO: Implement loading
        return registry
