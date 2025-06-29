# /src/data/sample_entries.py

from uuid import uuid4
from src.config import FAISS_VECTOR_DIM


SAMPLE_VECTORS = [
    {
        "id": str(uuid4()),
        "vector": [0.1 * i for i in range(FAISS_VECTOR_DIM)],
        "metadata": {
            "title": "The Matrix",
            "year": 1999,
            "genres": ["Science Fiction", "Action"],
            "tagline": "Welcome to the Real World",
            "overview": "A hacker discovers the world is a simulation...",
            "actors": ["Keanu Reeves", "Laurence Fishburne"],
            "certificate": "R",
            "media_type": "Movie",
            "language": "English"
        }
    },
    {
        "id": str(uuid4()),
        "vector": [0.2 * i for i in range(FAISS_VECTOR_DIM)],
        "metadata": {
            "title": "Pulp Fiction",
            "year": 1994,
            "genres": ["Crime", "Drama"],
            "tagline": "Just because you are a character doesn't mean you have character.",
            "overview": "The lives of two mob hitmen, a boxer, and others intertwine.",
            "actors": ["John Travolta", "Samuel L. Jackson"],
            "certificate": "R",
            "media_type": "Movie",
            "language": "English"
        }
    }
]

def get_sample_entries():
    sample_entries = [
  {
    "title": "Samurai Rauni",
    "year": 2016,
    "genres": [
      "Comedy",
      "Drama"
    ],
    "overview": "Villagers are afraid of Samurai Rauni Reposaarelainen, who keeps them on their toes every day. When someone places a bounty on Rauni's head, he goes after this mysterious person.",
    "tagline": "Who would move a mountain, if not the mountain itself.",
    "certificate": "",
    "actors": [
      "Mika Ratto",
      "Reetta Turtiainen",
      "Veera Elo",
      "Harri Sippola",
      "Juha Hurme",
      "Risto Yliharsila"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Remote Control",
    "year": 1988,
    "genres": [
      "Horror",
      "Science Fiction",
      "Comedy"
    ],
    "overview": "A video store clerk stumbles onto an alien plot to take over earth by brainwashing people with a bad '50s science fiction movie. He and his friends race to stop the aliens before the tapes can be distributed world-wide.",
    "tagline": "Your future is in their hands.",
    "certificate": "R",
    "actors": [
      "Kevin Dillon",
      "Deborah Goodrich",
      "Jennifer Tilly",
      "Frank Beddor",
      "Christopher Wynne",
      "Kaaren Lee",
      "Bert Remsen",
      "Jamie McEnnan",
      "Jerold Pearson",
      "Jennifer Buchanan",
      "Marilyn Adams",
      "Dick Warlock",
      "John Lafayette",
      "Deborah Downey",
      "Lisa Aliff"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Cassius X: Becoming Ali",
    "year": 2023,
    "genres": [
      "Documentary"
    ],
    "overview": "Cassius X puts a period of often-overlooked history into the spotlight  the period when Cassius Clay fought his way to achieving his lifelong dream of becoming World Heavyweight Champion while embarking on a secret spiritual journey.",
    "tagline": "",
    "certificate": "12A",
    "actors": [
      "Muhammad Ali",
      "Jim Lampley",
      "Thomas Hauser",
      "Robert Lipsyte",
      "Jerry Izenberg",
      "Attallah Shabazz",
      "Dee Dee Sharp"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Are You Being Served? Secrets & Scandals",
    "year": 2022,
    "genres": [
      "Documentary",
      "Comedy"
    ],
    "overview": "Are You Being Served? is one of the most popular and most outrageous British sitcoms of all time. For more than a decade, millions of viewers tuned in for its smutty innuendo and the electric chemistry between the cast. But behind the laughter were plenty of secrets and scandals.",
    "tagline": "",
    "certificate": "",
    "actors": [
      "Kate Robbins",
      "Trevor Bannister",
      "Mollie Sugden",
      "Harold Bennett",
      "John Inman",
      "Wendy Richard",
      "Nicholas Smith",
      "Frank Thornton"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Cockaboody [Kurzfilm]",
    "year": 1973,
    "genres": [
      "Animation",
      "Short",
      "k.A."
    ],
    "overview": "John and Faith Hubley combined animation with the voices of their preschool daughters Georgia and Emily to make this award-winning short (New York Animation Festival), similar in concept to their earlier work \"Moonbird\".",
    "tagline": "",
    "certificate": "",
    "actors": [],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "King, Queen, Joker",
    "year": 1921,
    "genres": [
      "Comedy"
    ],
    "overview": "King, Queen, Joker is a 1921 silent feature farce written and directed by Sydney Chaplin, Charlie's older brother. The picture was produced by Famous Players-Lasky and distributed through Paramount Pictures. The film was shot in England, France and the United States.  Less than a reel of this film, the barbershop sequence, survives at the British Film Institute. It was included in the 2011 Criterion DVD special two disc edition release of The Great Dictator.",
    "tagline": "",
    "certificate": "",
    "actors": [
      "Sydney Chaplin",
      "Lottie MacPherson"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Origin of the Species",
    "year": 2020,
    "genres": [
      "Documentary"
    ],
    "overview": "Origin of the Species is an experimental documentary that explores the current climate of android development with a focus on human/machine relations, gender and the ethical implications of this research. The film provides an insider look into cutting edge laboratories in Japan and the USA where scientists attempt to make robots move, speak and look human. These scientists and their discoveries are contextualized with cinematic and pop culture references, to underline the mythic, comic and uncanny aspects of our aspiration to create machines that are eerily similar to ourselves.",
    "tagline": "Is a new era dawning for robotics?",
    "certificate": "",
    "actors": [],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "The Parrotville Fire Department",
    "year": 1934,
    "genres": [
      "Animation"
    ],
    "overview": "2nd Cartoon in the Rainbow Parade Series by Van Beuren.",
    "tagline": "",
    "certificate": "",
    "actors": [],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "The World According to Amazon",
    "year": 2019,
    "genres": [
      "Documentary"
    ],
    "overview": "This film dives into the world of Amazon, its story and view of the world. It offers a large social fresco backed up by an in-depth investigation where private lives meet the mega-machine.",
    "tagline": "",
    "certificate": "",
    "actors": [],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Operation Jericho",
    "year": 2011,
    "genres": [
      "Documentary",
      "History",
      "War"
    ],
    "overview": "Actor and aviator Martin Shaw takes to the skies to rediscover one of the most audacious and daring raids of World War II.  On the morning of 18th February 1944, a squadron of RAF Mosquito bombers, flying as low as three metres over occupied France, demolished the walls of Amiens Jail in what became known as Operation Jericho. The reasons behind the controversial raid remain a mystery to this day.  This dramatic documentary investigates the missing pieces of the story, with interviews from survivors and aircrew, and tries to find out why the raid was ordered and by whom.",
    "tagline": "",
    "certificate": "",
    "actors": [
      "Martin Shaw"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Search For The Edge Of Space",
    "year": 2024,
    "genres": [
      "Documentary"
    ],
    "overview": "The universe has long captivated us with its immense scales of distance and time. Many astronomers today have come to believe that what we can see represents only a small fraction of all there is. They are pioneering bold new theories that describe a cosmic landscape that extends far beyond the limits of our vision. What lies beyond the streams of galaxies that extend as far as our telescopes can see? Where does it all end? How do we fit within it?",
    "tagline": "",
    "certificate": "",
    "actors": [],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "2",
    "year": 2009,
    "genres": [],
    "overview": "",
    "tagline": "",
    "certificate": "",
    "actors": [
      "Linda Soto",
      "Eduardo Quispe"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Miners Strike The Battle For Britain",
    "year": 1984,
    "genres": [],
    "overview": "",
    "tagline": "",
    "certificate": "",
    "actors": [],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Penguin Bloom",
    "year": 2021,
    "genres": [
      "Drama"
    ],
    "overview": "When an unlikely ally enters the Bloom family's world in the form of an injured baby magpie they name Penguin, the birds arrival makes a profound difference in the struggling familys life.",
    "tagline": "The true story of an unlikely hero.",
    "certificate": "Rated PG-13",
    "actors": [
      "Naomi Watts",
      "Andrew Lincoln",
      "Griffin Murray-Johnston",
      "Felix Cameron",
      "Abe Clifford-Barr",
      "Jacki Weaver",
      "Rachel House",
      "Gia Carides",
      "Leeanna Walsman",
      "Lisa Hensley",
      "Randolph Fields"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "The Joy of Stats",
    "year": 2010,
    "genres": [
      "Documentary"
    ],
    "overview": "Professor Hans Rosling shares his excitement with statistics, and shows how researchers are handling the modern data deluge.",
    "tagline": "",
    "certificate": "",
    "actors": [
      "Hans Rosling"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "A Brief History of Time Travel",
    "year": 2018,
    "genres": [
      "Documentary"
    ],
    "overview": "A journey through the evolution of time travel; from it origins, it's evolution and influence in science fiction, to the exciting possibilities in the future.",
    "tagline": "",
    "certificate": "",
    "actors": [
      "Ted Chiang",
      "Satyanarayana Dasa",
      "Dr. Erik D. Demaine",
      "Ronald Mallett",
      "Bill Nye",
      "Brooks Peck",
      "Michelle Power",
      "Tim Schafer",
      "Seth Shostak"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Vysoke Tatry",
    "year": 1966,
    "genres": [
      "Documentary",
      "Short"
    ],
    "overview": "Documentary about a mountain range and surrounding area.",
    "tagline": "",
    "certificate": "",
    "actors": [],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Madame's Cravings",
    "year": 1907,
    "genres": [
      "Comedy",
      "Short"
    ],
    "overview": "A heavily pregnant woman has a series of irrepressible cravings while walking with her family.",
    "tagline": "",
    "certificate": "",
    "actors": [],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Harley Quinn S01e13",
    "year": 2020,
    "genres": [],
    "overview": "",
    "tagline": "",
    "certificate": "",
    "actors": [],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Bubble Bubble Bubble Meows",
    "year": 2022,
    "genres": [
      "Animation",
      "Comedy",
      "Family"
    ],
    "overview": "A poorly drawn cat, the end of the world, and gum.",
    "tagline": "",
    "certificate": "",
    "actors": [
      "Briana Bui",
      "Brooke deRosa",
      "S. Joe Downing",
      "Ian Lyons",
      "Ogechukwu Ebi",
      "Travis Doane"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Would I Lie To You At Christmas (2024)",
    "year": 2024,
    "genres": [],
    "overview": "",
    "tagline": "",
    "certificate": "",
    "actors": [],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "The Simpsons: The Past and the Furious",
    "year": 2025,
    "genres": [
      "Animation",
      "Comedy",
      "TV Movie",
      "Science Fiction"
    ],
    "overview": "Lisa travels back in time to 1923 and discovers that the Springfield Mini Moose, once key to the towns ecosystem, were driven to extinction in 1925. Teaming up with young Monty Burns, Lisa works to save the moose, but her actions unintentionally shape his future as a ruthless tycoon.",
    "tagline": "",
    "certificate": "",
    "actors": [
      "Yeardley Smith",
      "Harry Shearer",
      "Dan Castellaneta",
      "Julie Kavner",
      "Nancy Cartwright",
      "Joseph Gordon-Levitt",
      "Hank Azaria",
      "Pamela Hayden",
      "Tress MacNeille",
      "Kevin Michael Richardson",
      "Alex Desert",
      "Chris Edgerly",
      "Maggie Roswell",
      "Dawnn Lewis"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Ghoulies",
    "year": 1984,
    "genres": [
      "Comedy",
      "Fantasy",
      "Horror"
    ],
    "overview": "A young man and his girlfriend move into an old mansion home, where he becomes possessed by a desire to control ancient demons.",
    "tagline": "They'll get you in the end.",
    "certificate": "PG-13",
    "actors": [
      "Peter Liapis",
      "Lisa Pelikan",
      "Michael Des Barres",
      "Jack Nance",
      "Peter Risch",
      "Tamara De Treaux",
      "Scott Thomson",
      "Ralph Seymour",
      "Mariska Hargitay",
      "Victoria Catlin",
      "Bobbie Bresee",
      "Keith Joe Dick",
      "David Dayan",
      "Charene Cathleen",
      "Jamie Bronow"
    ],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Mechanical Principles",
    "year": 1930,
    "genres": [
      "Documentary"
    ],
    "overview": "Close up we see pistons move up and down or side to side. Pendulums sway, the small parts of machinery move. Gears drive larger wheels. Gears within gears spin. Shafts turn some mechanism that is out of sight. Screws revolve and move other gears; a bit rotates. More subtle mechanisms move other mechanical parts for unknown purposes. Weights rise and fall. The movements, underscored by sound, are rhythmic. Circles, squares, rods, and teeth are in constant and sometimes asymmetrical motion. These human-made mechanical bits seem benign and reassuring.",
    "tagline": "",
    "certificate": "",
    "actors": [],
    "media_type": "Movie",
    "language": "English"
  },
  {
    "title": "Spinning a Yarn: The Dubious History of Scottish Tartan",
    "year": 2013,
    "genres": [
      "Documentary"
    ],
    "overview": "Moray Hunter narrates the story of tartan's murky past and colourful present, taking in the Englishmen who forged a guide to clan tartans, Walter Scott's tartan pageant of 1822 and the 21st-century Scottish Register of Tartans.",
    "tagline": "",
    "certificate": "",
    "actors": [
      "Brian Wilton",
      "Andy Taylor",
      "Fiona Anderson"
    ],
    "media_type": "Movie",
    "language": "English"
  }
]

    # Ensure each entry has a 'document' field
    for entry in sample_entries:
        entry["document"] = f"{entry['title']} ({entry['year']})"

    return sample_entries


def get_sample_vectors():
    """Full ingestion entries with vectors."""
    return SAMPLE_VECTORS


def get_formatter_sample():
    """Only the rich metadata of the first movie for formatter tests."""
    return SAMPLE_VECTORS[0]["metadata"]
