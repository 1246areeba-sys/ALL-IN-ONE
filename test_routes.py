import urllib.request, urllib.parse, json

base = 'http://127.0.0.1:5000'
out = []
for r in ['/', '/house', '/spotify', '/analyze']:
    try:
        s = urllib.request.urlopen(base + r, timeout=30).status
        out.append(f'{r} -> {s}')
    except Exception as e:
        out.append(f'{r} -> ERROR {e}')

# house predict
data = urllib.parse.urlencode({'area_sqm': '200', 'rate_per_sqft': '20000',
                               'bedRoom_n': '3', 'bathroom_n': '2',
                               'balcony_n': '2', 'noOfFloor_n': '2'}).encode()
req = urllib.request.Request(base + '/house/predict', data=data, method='POST')
out.append('house predict -> ' + urllib.request.urlopen(req, timeout=30).read().decode())

# spotify predict
data = urllib.parse.urlencode({'danceability': '0.6', 'energy': '0.7', 'tempo': '120',
                               'loudness': '-7', 'valence': '0.5', 'acousticness': '0.3',
                               'duration_ms': '200000', 'playlist_genre': 'pop',
                               'playlist_subgenre': 'mainstream'}).encode()
req = urllib.request.Request(base + '/spotify/predict', data=data, method='POST')
out.append('spotify predict -> ' + urllib.request.urlopen(req, timeout=30).read().decode())

with open('route_test.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('\n'.join(out))
