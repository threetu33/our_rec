from openai import OpenAI
client = OpenAI(base_url="http://115.182.62.174:18888/v1", api_key="sk-ZUK05LeE0U3FVQAp6e5eCd3628774c4181D513786bD3B901")

model_names = ["deepseek-r1-0528"]

prompt = f"""
You are a recommendation system for Musical products.\n\nUser's purchase history: Portable Stand for Acoustic and Classical Guitars by Hola! Music -> WINGO Wide Guitar Capo Fit for 6 and 12 String Acoustic Classical Electric Guitar,Bass,Mandolin,Banjos,Ukulele All Types String Instrument, Black\n\nPlease rank the following 20 items by likelihood of purchase.\n\nCandidate items:\n1. Gruv Gear FretWraps String Muters 3-Pack (Wood Print, Medium) (FW-3PK-WOD-MD)\n2. LOUDmouth Dual Wireless Microphone Pouch | Zippered Mic Bag for Two Long Microphones | 12.5\" x 6.5\n3. Drum Keys 3 Pack Universal Drum Tuning Key with Continuous Motion Speed Key\n4. Pig Hog PHM20BKW Black/White Woven High Performance XLR Microphone Cable, 20 Feet\n5. Guitar Wall Mount Hangers, Guitar Wall Hanger 2 Pack,Guitar Hanger Wall Mount for Home and Studio\n6. Focusrite Saffire Pro 14 8 In / 6 Out FireWire Audio Interface with 2 Focusrite Mic Preamps\n7. KAISH Bass Guitar String Tree Bass String Retainer with Mouting Screws Chrome\n8. FLAMMA 9V 300mA Power Supply for Electric Guitar Effects Pedals Tip Negative\n9. Snark Super Snark HZ Clip-On Tuner - Tunes Guitar, Bass and All Instruments\n10. Hosco 3 Guitar Nut File set TL-NF3E String Slots Electric Guitar VWWS Japan Ships From USA\n11. Yovus XLR Male to Female 3pin Mic Microphone Lo-z Extension Cable Cord (10 Foot Feet ft, Blue)\n12. Piano Stickers for Keys   Removable w/Double Layer Coating for 49/61 / 76/88 Keyboards\n13. D Addario Woodwinds Soprano Sax Reeds, Strength 1.5, 3-pack\n14. Tama Speed Cobra 310 Hi-hat Stand\n15. On-Stage MB7006 6-Space Microphone and Accessory Bag\n16. Gator Frameworks ID Series Speaker Stand Set with Padded Nylon Carry Bag; Set of 2 Stands (GFW-ID-SPKRSET),Black\n17. Allparts Tremol-No Tremolo Locking Device - Pin Type\n18. Zoom B3n Bass Guitar Multi-Effects Processor Pedal, With 60+ Built-in effects, Amp Modeling, Stereo Effects, Looper, Rhythm Section, Tuner\n19. Fender American Series Locking Strap Buttons - Chrome\n20. ART CoolSwitchPro Isolated A/B-Y Switch Instrument Pedal with Footswitch\n\nIMPORTANT: Your response must end with exactly one line in this format:\nRANKING: number1,number2,number3,number4,number5,number6,number7,number8,number9,number10,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20\n\nWhere each number is between 1-20. Example:\nRANKING: 26,45,78,50,38,99,77,43,53,89,8,93,97,52,47,31,48,83,98,79\n
"""

response = client.chat.completions.create(
    model=model_names[0],
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": prompt}
    ]
)
print("🗼🌍✈️thinking🗼🌍✈️")
print(response.choices[0].message.reasoning)
print("🗼🌍✈️response🗼🌍✈️")
print(response.choices[0].message.content)