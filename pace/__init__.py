
ID_TO_BEAT = {0:['N','L','R','e','j'], 1:['S','A','a','J'], 2:['V','E'], 3:['F'], 4:['/','Q','f']}
ID_TO_AAMI = {0:'Normal', 1:'Supraventricular premature', 2:'Premature ventricular contraction', 3:'Fusion of ventricular & normal', 4:'Unclassifiable'}

PATIENT_IDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]

BEAT_TO_ID = {}

for ID, beats in ID_TO_BEAT.items():
        for beat in beats:
            BEAT_TO_ID[beat] = ID
