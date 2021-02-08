# Repository QTI Final Project

Dataset "News Classification: Inshort daily news data"
from https://www.kaggle.com/kishanyadav/inshort-news


Additional features: 
- Inputs and outputs of the app are stored in a PostgreSQL database                     
- Some routes to implement CRUD for the database

Development stage:
- Run this code to create PostgreSQL database container: 
  - docker run --name pg_news -p 5432:5432 -e POSTGRES_PASSWORD=password123 -d postgres
- Run the app
