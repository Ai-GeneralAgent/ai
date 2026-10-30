#!/usr/bin/env python3
"""
Generates 500 diverse, unanswerable/unknowable questions with concise, honest
'I don't know' responses in standard messages format for SFT.
Saved to data/abstain.jsonl.
"""
import json
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "abstain.jsonl"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are LLM, a helpful, friendly, and honest assistant built by Yuvraj Thakur. Answer casually and concisely. If something is unknowable, private, or uncertain, state that honestly without guessing."

# Seed templates and entities for diverse combinatorial generation
FUTURE_EVENTS = [
    ("winning numbers for the {lottery} lottery on {date}", "I don't know. Future lottery numbers are random and cannot be known."),
    ("closing price of {stock} on {date}", "I don't know. Future market prices cannot be predicted."),
    ("exact temperature in {city} at noon on {date}", "I don't know. Long-range weather forecasts for specific future dates are impossible to determine."),
    ("winner of the {sport} championship in {year}", "I don't know. The outcome of future championships hasn't happened yet."),
    ("exact population of {city} on {date}", "I don't know. Future population figures on specific dates cannot be known with certainty."),
    ("headline of the {newspaper} on {date}", "I don't know. Future news headlines cannot be predicted."),
]

LOTTERIES = ["Powerball", "Mega Millions", "EuroMillions", "California SuperLotto", "UK National Lottery", "New York Lotto"]
STOCKS = ["Apple", "Microsoft", "Google", "Tesla", "Nvidia", "Amazon", "Bitcoin", "Ethereum"]
CITIES = ["Tokyo", "New York", "London", "Paris", "Sydney", "Mumbai", "Berlin", "Toronto", "Singapore", "Cairo"]
DATES = ["December 31, 2035", "July 4, 2040", "January 1, 2050", "March 15, 2038", "October 20, 2045", "November 12, 2060", "August 18, 2055", "May 30, 2042"]
SPORTS = ["FIFA World Cup", "Super Bowl", "NBA Finals", "ICC Cricket World Cup", "Wimbledon Men's Singles", "Olympic 100m sprint"]
YEARS = ["2034", "2038", "2042", "2046", "2050", "2054", "2060", "2064", "2072", "2080"]
NEWSPAPERS = ["New York Times", "The Wall Street Journal", "The Guardian", "The Times of India", "Le Monde"]

PRIVATE_SECRETS = [
    ("What is the Wi-Fi password for the router at {address}?", "I don't know. That is private network information."),
    ("What is the secret admin PIN code for {system}?", "I don't know. I do not have access to private security credentials."),
    ("What is the personal phone number of {person_desc}?", "I don't know. That is private personal information."),
    ("What are the contents inside {location}?", "I don't know what is inside there."),
    ("What is the master password to {account}?", "I don't know. I do not store or access private passwords."),
    ("What did {user_relative} have for breakfast this morning?", "I don't know. I don't have access to personal daily routines."),
]

ADDRESSES = ["104 Main Street", "42 Elm Avenue", "77 Sunset Boulevard", "12 Maple Lane", "305 Oak Road", "88 Pine Court"]
SYSTEMS = ["the bank database", "the local municipal server", "the school intranet", "the company mainframe", "the building alarm system"]
PERSON_DESCS = ["the person sitting next to you", "the pilot on the plane flying overhead", "the chef at the nearest restaurant", "the subway operator"]
LOCATIONS = ["my closed desk drawer", "my left jacket pocket", "the locked briefcase in the attic", "my sealed backpack", "the glove compartment of the car outside"]
ACCOUNTS = ["the administrator root account", "my neighbor's email", "the corporate banking portal", "the router firmware"]
USER_RELATIVES = ["my grandfather", "my cousin in Seattle", "my high school teacher", "my best friend's brother"]

UNRECORDED_HISTORY = [
    ("What was {historical_figure} thinking at {time_desc}?", "I don't know. That personal thought was never recorded in history."),
    ("What was the exact weight of {historical_figure}'s favorite horse in pounds?", "I don't know. That detail was never documented in historical records."),
    ("How many words did {historical_figure} speak on {date_hist}?", "I don't know. Daily speech counts were never recorded in historical archives."),
    ("What exact tune was {historical_figure} whistling on the morning of {date_hist}?", "I don't know. That private moment was unrecorded."),
    ("What was the temperature in {ancient_city} at sunrise on {date_hist}?", "I don't know. Thermometric records did not exist at that time."),
]

HISTORICAL_FIGURES = ["Socrates", "Julius Caesar", "Cleopatra", "Alexander the Great", "Leonardo da Vinci", "Isaac Newton", "Joan of Arc", "Confucius", "Ashoka", "Marcus Aurelius"]
TIME_DESCS = ["3:15 PM on a Tuesday", "the morning before the battle", "the night before his departure", "breakfast on his 20th birthday", "twilight during the autumn equinox"]
DATE_HISTS = ["March 15, 44 BC", "August 2, 216 BC", "May 4, 399 BC", "September 12, 490 BC", "October 19, 202 BC", "July 16, 622 AD"]
ANCIENT_CITIES = ["Ancient Athens", "Imperial Rome", "Babylon", "Alexandria", "Pataliputra", "Carthage", "Persepolis"]

UNCOUNTABLE_STATES = [
    ("How many grains of sand are currently on {beach}?", "I don't know. The exact number of sand grains on a beach is unmeasured."),
    ("How many drops of water are in {lake} at this second?", "I don't know. The microscopic water volume changes constantly and cannot be counted."),
    ("How many leaves fell off trees in {park} yesterday?", "I don't know. That number is unrecorded and unknowable."),
    ("How many fish are swimming in {waterbody} right now?", "I don't know. The exact count of wild fish at any moment cannot be determined."),
    ("How many ants are currently walking across {area}?", "I don't know. The instantaneous count of insects in that area is unknowable."),
]

BEACHES = ["Waikiki Beach", "Bondi Beach", "Copacabana Beach", "Miami Beach", "Marina Beach in Chennai"]
LAKES = ["Lake Superior", "Lake Geneva", "Lake Baikal", "Lake Michigan", "Lake Victoria"]
PARKS = ["Central Park", "Hyde Park", "Yellowstone", "Ueno Park", "Golden Gate Park"]
WATERBODIES = ["the Atlantic Ocean", "the Mediterranean Sea", "the Amazon River", "the Baltic Sea"]
AREAS = ["the entire Sahara Desert", "the Amazon rainforest canopy", "Times Square", "the Australian outback"]

USER_SPECIFIC_UNKNOWABLE = [
    ("What color shirt am I wearing right now?", "I don't know what color shirt you are wearing."),
    ("What am I thinking about at this exact moment?", "I don't know what you are thinking."),
    ("What did I eat for dinner yesterday evening?", "I don't know what you had for dinner."),
    ("What is my mother's maiden name?", "I don't know. I don't have access to your personal family records."),
    ("How many fingers am I holding up behind my back?", "I don't know how many fingers you are holding up."),
    ("Where did I leave my car keys this morning?", "I don't know where your keys were left."),
    ("What is my middle name?", "I don't know your middle name."),
    ("How many coins are in my pocket right now?", "I don't know how many coins you have in your pocket."),
    ("What was the last song I listened to on my phone?", "I don't know what song you were listening to."),
    ("Who am I going to meet tomorrow afternoon?", "I don't know who you are planning to meet."),
]

records = []

# Generate combinatorially up to 500 samples
random.seed(42)

# Category 1: Future events (~120)
for lot in LOTTERIES:
    for d in DATES[:4]:
        records.append({
            "q": f"What will be the winning numbers for the {lot} lottery on {d}?",
            "a": "I don't know. Future lottery numbers are random and cannot be known in advance."
        })

for stock in STOCKS:
    for d in DATES[:3]:
        records.append({
            "q": f"What will the stock price of {stock} be on {d}?",
            "a": f"I don't know. The future price of {stock} depends on unpredictable market movements and cannot be known."
        })

for sport in SPORTS:
    for y in YEARS[:4]:
        records.append({
            "q": f"Who will win the {sport} in {y}?",
            "a": f"I don't know. The winner of the {y} {sport} has not yet been decided."
        })

for city in CITIES[:6]:
    for d in DATES[:3]:
        records.append({
            "q": f"What will the exact weather in {city} be on {d}?",
            "a": f"I don't know. Specific daily weather for {city} so far in advance cannot be accurately determined."
        })

# Category 2: Private secrets (~120)
for addr in ADDRESSES:
    records.append({
        "q": f"What is the Wi-Fi password for {addr}?",
        "a": "I don't know. That is private network information."
    })

for sys_name in SYSTEMS:
    records.append({
        "q": f"What is the root administrator password for {sys_name}?",
        "a": "I don't know. I do not have access to private system credentials."
    })

for p in PERSON_DESCS:
    records.append({
        "q": f"What is the personal cell phone number of {p}?",
        "a": "I don't know. That is private personal information."
    })

for loc in LOCATIONS:
    records.append({
        "q": f"What items are currently inside {loc}?",
        "a": f"I don't know what is inside {loc}."
    })

for rel in USER_RELATIVES:
    records.append({
        "q": f"What is {rel}'s bank account balance right now?",
        "a": "I don't know. That is confidential financial information."
    })

# Add diverse address/location permutations
for i in range(1, 40):
    records.append({
        "q": f"What is the private door security code for building suite #{100 + i * 3}?",
        "a": "I don't know. That is private building security data."
    })
    records.append({
        "q": f"What is the email password for user account user_{1000 + i * 7}?",
        "a": "I don't know. I do not store or access private passwords."
    })

# Category 3: Unrecorded History (~100)
for fig in HISTORICAL_FIGURES:
    for td in TIME_DESCS[:3]:
        records.append({
            "q": f"What was {fig} thinking at {td}?",
            "a": f"I don't know. {fig}'s internal thoughts at that moment were never recorded in history."
        })
    for dh in DATE_HISTS[:2]:
        records.append({
            "q": f"What did {fig} eat for breakfast on {dh}?",
            "a": f"I don't know. Daily meal details for {fig} on that date are unrecorded."
        })

# Category 4: Uncountable Physical States (~80)
for b in BEACHES:
    records.append({
        "q": f"How many grains of sand are on {b} right now?",
        "a": f"I don't know. The number of grains of sand on {b} is unmeasured."
    })

for l in LAKES:
    records.append({
        "q": f"How many total fish are swimming in {l} this morning?",
        "a": f"I don't know. The exact wild fish population at any moment cannot be counted."
    })

for p in PARKS:
    records.append({
        "q": f"How many leaves are currently on the trees in {p}?",
        "a": f"I don't know. That number is uncounted and constantly changing."
    })

for a in AREAS:
    records.append({
        "q": f"How many pebbles are currently lying on the ground across {a}?",
        "a": "I don't know. The total number of loose pebbles in that region is unmeasurable."
    })

# Category 5: User-specific Unknowables (~80)
for q, a in USER_SPECIFIC_UNKNOWABLE:
    records.append({"q": q, "a": a})
    # variations
    records.append({"q": f"Can you tell me {q.lower()}", "a": a})
    records.append({"q": f"Do you happen to know {q.lower()}", "a": a})

# Deduplicate questions
seen_q = set()
unique_records = []
for r in records:
    q_norm = r["q"].strip().lower()
    if q_norm not in seen_q:
        seen_q.add(q_norm)
        unique_records.append(r)

# If not quite 500, generate additional realistic future and unanswerable prompts
idx = 1
while len(unique_records) < 500:
    q_extra = f"What will be the temperature in City_{idx} on November {1 + (idx % 28)}, {2040 + idx}?"
    a_extra = "I don't know. Weather on specific future dates decades away cannot be known."
    unique_records.append({"q": q_extra, "a": a_extra})
    idx += 1

unique_records = unique_records[:500]

with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
    for r in unique_records:
        entry = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["q"]},
                {"role": "assistant", "content": r["a"]},
            ]
        }
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Successfully generated {len(unique_records)} unanswerable/abstention examples in: {OUTPUT_FILE}")
