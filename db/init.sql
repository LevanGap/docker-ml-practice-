CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    input_features TEXT NOT NULL,
    prediction TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
