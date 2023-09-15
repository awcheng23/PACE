
ID_to_beat = {0:['N','L','R','e','j'], 1:['S','A','a','J'], 2:['V','E'], 3:['F'], 4:['/','Q','f']}
ID_to_AAMI = {0:'Normal', 1:'Supraventricular premature', 2:'Premature ventricular contraction', 3:'Fusion of ventricular & normal', 4:'Unclassifiable'}

beat_to_ID = {}

for ID, beats in ID_to_beat.items():
        for beat in beats:
            beat_to_ID[beat] = ID
