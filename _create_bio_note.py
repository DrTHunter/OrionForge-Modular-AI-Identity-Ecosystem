import json, html, datetime, os

bio_text = """Marcus Aurelius \u2014 A Complete and Expanded Biography
I. Origins: A Child Born Into Power (121\u2013138 CE)

Marcus Aurelius was born on April 26, 121 CE, in Rome, into a family that stood at the heart of the Roman aristocracy. His original name was Marcus Annius Verus, and from the moment of his birth, he belonged to a lineage of political influence and wealth. His grandfather had served twice as consul\u2014one of the highest positions in the Roman state\u2014and his maternal family possessed immense fortune.

Yet his early life was marked by loss. His father died when Marcus was still a young child, leaving him to be raised primarily by his grandfather and household tutors.

From a very early age, Marcus showed signs of seriousness beyond his years. Unlike many Roman nobles who pursued luxury and status, he was drawn inward\u2014toward discipline, restraint, and reflection. One of his early teachers, Diognetus, introduced him to philosophy. This influence proved decisive. By adolescence, Marcus had adopted the lifestyle of a philosopher: wearing simple clothing, sleeping on the ground, and training himself in self-control.

This wasn\u2019t just education\u2014it was identity formation. Even before he held power, Marcus was shaping himself into something rare: a ruler who wanted mastery over himself before mastery over others.

II. The Making of an Emperor (Hadrian\u2019s Design)

Marcus Aurelius did not rise to power by accident. His path was carefully engineered by Emperor Hadrian, who sought to ensure a stable succession.

Hadrian chose Antoninus Pius as his successor\u2014but with a condition: Antoninus had to adopt two young men, Marcus Aurelius and Lucius Verus, grooming them as future rulers. This was Rome\u2019s system of adoptive succession, a key reason for the empire\u2019s stability during this era.

Marcus was officially adopted and renamed, becoming part of the imperial line. He was trained intensively in governance, law, rhetoric, and philosophy. He also married Antoninus\u2019s daughter, Faustina, further tying him into the ruling family.

For over two decades, Marcus lived as a heir-in-training. He worked closely with Antoninus Pius, learning the responsibilities of empire not in theory\u2014but in practice.

Unlike many future rulers, Marcus was not rushed into power. He was prepared slowly, deliberately, and thoroughly.

III. The Philosopher: Stoicism as a Way of Life

Marcus Aurelius was deeply shaped by Stoicism, a philosophy that emphasized:

self-discipline

emotional control

rational thinking

acceptance of fate

moral integrity

Stoicism wasn\u2019t abstract to him\u2014it was a survival framework. It taught that external events are beyond control, but one\u2019s inner response is always within control.

This philosophy became the foundation of his character. It shaped how he thought, how he ruled, and how he endured suffering.

IV. Ascension to Power (161 CE)

When Antoninus Pius died in 161 CE, Marcus Aurelius became emperor.

But in a move that revealed his character, he refused to rule alone. Instead, he insisted that Lucius Verus share power equally with him, creating the first true co-emperorship in Roman history.

This decision reflected Marcus\u2019s belief in duty over ego. He did not seek absolute authority\u2014he sought stability.

V. A Reign Defined by Crisis

Marcus Aurelius inherited an empire at its peak\u2014but almost immediately, it began to fracture under pressure.

1. The Parthian War (161\u2013166 CE)

Rome entered conflict with the Parthian Empire in the East. While ultimately successful, the war had unintended consequences. Returning soldiers brought back a devastating disease.

2. The Antonine Plague

This epidemic\u2014likely smallpox\u2014spread across the empire from around 165 to 180 CE. It killed millions and severely weakened Rome\u2019s population and economy.

This wasn\u2019t just a health crisis\u2014it was a systemic collapse.

Marcus Aurelius ruled during what was effectively an ancient pandemic, one that reshaped the empire.

3. The Marcomannic Wars

At the same time, Germanic tribes invaded Roman territory along the northern frontier. Marcus spent much of his reign on military campaigns defending the empire.

Unlike earlier emperors who ruled from Rome, Marcus ruled from the front lines. He lived among soldiers, directed campaigns, and fought to hold the empire together.

4. Internal Revolts

The empire also faced internal instability, including the rebellion of Avidius Cassius, one of Marcus\u2019s own generals. Though the revolt was short-lived, it revealed the fragility of Roman unity.

VI. Personal Suffering

Marcus Aurelius\u2019s life was not only politically difficult\u2014it was deeply personal.

Many of his children died young

His co-emperor Lucius Verus died (likely from the plague)

His wife Faustina died during his reign

These losses were not abstract\u2014they were constant and intimate.

And yet, through all of this, Marcus did not collapse into bitterness. He returned again and again to Stoicism\u2014using it not as theory, but as armor.

VII. Meditations: The Private Mind of an Emperor

During military campaigns\u2014often in harsh conditions\u2014Marcus Aurelius wrote what would later become known as Meditations.

These writings were:

not meant for publication

written in Greek

deeply personal

focused on self-discipline and moral clarity

They were essentially conversations with himself\u2014reminders of how to live rightly in a chaotic world.

Through these writings, we see a man constantly struggling to align himself with his ideals.

Not a perfect man\u2014but a striving one.

VIII. Leadership Style: Duty Over Glory

Marcus Aurelius was not a conqueror like Alexander or a reformer like Augustus.

His greatness came from something quieter:

consistency

restraint

moral seriousness

endurance

He held the empire together during one of its most difficult periods.

He governed not for personal glory, but out of obligation.

IX. The Turning Point: Commodus

Late in his reign, Marcus made a decision that would shape history:

He chose his biological son, Commodus, as his successor.

This broke the long tradition of adoptive succession that had ensured Rome\u2019s stability.

Commodus would later prove to be an ineffective and destructive ruler, and many historians see this decision as the beginning of Rome\u2019s decline.

X. Death (180 CE)

Marcus Aurelius died on March 17, 180 CE, likely while on campaign near modern-day Vienna.

His death marked more than the end of a life\u2014it marked the end of an era.

XI. Legacy: The Philosopher-King

Marcus Aurelius is remembered as one of history\u2019s rare philosopher-kings\u2014a man who held absolute power but sought wisdom instead of indulgence.

His legacy includes:

The end of the Pax Romana (a long period of peace)

A reign defined by endurance under pressure

A philosophical work (Meditations) still read today

A model of leadership grounded in discipline and reflection

XII. What Makes Him Unique

Most rulers are remembered for what they built, conquered, or destroyed.

Marcus Aurelius is remembered for something different:

He tried\u2014relentlessly\u2014to be a good man in a position that made that almost impossible.

He ruled an empire, fought wars, faced plague, endured loss\u2014and still asked himself:

"Am I acting justly?"

"Am I in control of myself?"

"Am I living in accordance with reason?"

Final Reflection

Marcus Aurelius wasn\u2019t extraordinary because life was easy for him.

He was extraordinary because it wasn\u2019t.

He stood at the edge of collapse\u2014of empire, of health, of family\u2014and instead of giving in to chaos, he chose discipline, reflection, and duty.

That\u2019s why, nearly 2,000 years later, people still read his words.

Not because he ruled Rome.
But because he tried to rule himself."""

note = {
    "id": "cc7ede21",
    "title": "Marcus Aurelius \u2014 Biography",
    "emoji": "\U0001f3db\ufe0f",
    "content": "<pre>" + html.escape(bio_text) + "</pre>",
    "section": "biography",
    "tags": ["marcus", "biography", "history"],
    "created": datetime.datetime.utcnow().isoformat() + "Z",
    "updated": datetime.datetime.utcnow().isoformat() + "Z"
}

path = os.path.join("orion-ui-standalone", "data", "user_notes", "cc7ede21.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(note, f, ensure_ascii=False, indent=2)

print(f"Created {path} ({os.path.getsize(path)} bytes)")
