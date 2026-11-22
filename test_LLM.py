
examples = [
    {
        "question": "Which projects for client Northstar are currently under construction?",
        "context": "Project records: Project Atlas, client Northstar, status construction; Project Beacon, client Northstar, status design; Project Cedar, client Northstar, status construction.",
    },
    {
        "question": "What is the total fee budget of the ongoing projects for client Northstar?",
        "context": "Ongoing projects for client Northstar: Atlas, fee budget €120,000; Cedar, fee budget €85,000. Completed project Birch, fee budget €40,000.",
    },
    {
        "question": "Who is the project manager for project Atlas?",
        "context": "Project Atlas project manager: Sarah Peeters. Project engineer: Tom Janssens.",
    },
    {
        "question": "Which heat pumps have been specified and what are their capacities?",
        "context": "Mechanical specification M-204, revision 3: HP-01, 120 kW; HP-02, 80 kW. Both are air-source heat pumps serving the main building.",
    },
    {
        "question": "What is the design flow rate of the sanitary installation?",
        "context": "Sanitary design report S-101, page 18: peak design flow rate is 4.8 litres per second. The value applies to the full building installation.",
    },
    {
        "question": "Where are the EV charging stations located?",
        "context": "Electrical layout E-302, revision 2: EV charging stations are in basement level -1, parking bays P01-P12, near grid column C.",
    },
    {
        "question": "Which outstanding remarks from site reports remain unresolved?",
        "context": "Site report 2026-04-12, remark 7: seal the roof penetration, assigned to contractor Delta, unresolved. Site report 2026-04-19, remark 3: replace the damaged door closer, assigned to contractor Echo, unresolved. Remark 5 was closed on 2026-04-21.",
    },
    {
        "question": "What are the latest changes made to the electrical installation?",
        "context": "Electrical drawing E-110 revision 4 dated 2026-03-12: added a sub-distribution board in the east wing. Revision 3 dated 2026-02-20: relocated the main board. Revision 4 is the latest revision.",
    },
    {
        "question": "Which documents contain conflicting information regarding the capacity of HP-01?",
        "context": "Mechanical schedule M-204 revision 3 lists HP-01 at 120 kW. Equipment datasheet DS-HP01 revision 1 lists HP-01 at 110 kW. No approved revision resolving the difference is included in the retrieved documents.",
    },
    {
        "question": "Which previous projects contain a similar HVAC solution to project Atlas?",
        "context": "Project Atlas uses air-source heat pumps with low-temperature underfloor heating. Project Birch uses air-source heat pumps and low-temperature underfloor heating. Project Cedar uses gas boilers and radiator heating. The retrieved records identify Birch as matching Atlas's described solution.",
    },
    {
        "question": "In which documents is AHU-03 mentioned?",
        "context": "Search results: Mechanical schedule M-204, page 12, mentions AHU-03; Commissioning report C-018, page 7, mentions AHU-03; Fire strategy F-006 contains no mention of AHU-03.",
    },
    {
        "question": "Provide the source documents used to formulate this answer.",
        "context": "Relevant retrieved sources: Mechanical schedule M-204, revision 3, page 12; Equipment datasheet DS-HP01, revision 1, page 2. These documents contain the capacity information about HP-01.",
    },
]

from LLM_model import generate_answer

for index, example in enumerate(examples, start=1):
    print(f"Example {index}: {example['question']}")
    try:
        print(generate_answer(example['question'], example['context']))
    except Exception as error:
        print(f"Error: {error}")
    print()
