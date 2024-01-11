from Database_Classes import (
    Database_Parent,
)  # If you get en error with not being able to find subclasses
from Embedding_Classes import (
    Embedding_Parent,
)  # Change the imports to: from ..... import *
from Summerization_Classes import Summerization_Parent
from Visualization_Classes import Visualization_Parent

database_parent = Database_Parent()
embedding_parent = Embedding_Parent()
summerization_parent = Summerization_Parent()
visualization_parent = Visualization_Parent()

database_parent.run()  # Grabs latsest version that is not WIP
database_parent.run("1.0")  # Grabs specific version
