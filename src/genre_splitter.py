#!/usr/bin/env python3
# coding: utf-8
"""
genre_splitter.py
04-17-19
jack skrable
"""

import re
import numpy as np
from collections import Counter


def target_genre(row):

    try:
        row = np.concatenate([re.split('\W+', s) for s in row])
        c = Counter(row)
        genres = [w for w, c in c.most_common(3)]
    except Exception as e:
        genres = ['None']

    if any(i for i in genres if i in ['grime','hyphy','hip','hop','rap','crunk','turntablism',
                                      'gangster','conscious','ghetto','bay']):
        target = 'hip-hop'

    elif any(i for i in genres if i in ['techno','house','electronica','trance','dj',
                                        'electronic','dubstep','indietronica','rave',
                                        'acid','club','electro','edm','hardstyle','dance',
                                        'breakbeat','jungle','eurodance']):
        target = 'techno'

    elif any(i for i in genres if i in ['folk','neofolk','acoustic','songwriter']):
        target = 'folk'

    elif any(i for i in genres if i in ['reggae','reggaeton','dancehall','rasta',
                                        'caribbean','dub','jamaica']):
        target = 'reggae'

    elif any(i for i in genres if i in ['gospel','religious','ritual','spiritual',
                                        'christian','worship','hymns']):
        target = 'religious'

    elif any(i for i in genres if i in ['classical','classic','chamber','orchestra',
                                        'concerto','composer','string','opera',
                                        'symphony','baroque','score']):
        target = 'classical'

    elif any(i for i in genres if i in ['samba','salsa','latin','latino','flamenco',
                                        'merengue','ranchera','mambo','mariba','mariachi',
                                        'tango','charanga','cumbia','spanish']):
        target = 'latin'

    elif any(i for i in genres if i in ['experimental','avant','avantgarde','garde',
                                        'modern','art','contemporary']):
        target = 'avant-garde'

    elif any(i for i in genres if i in ['punk','ska','emo','rockabilly','sleaze',
                                        'hardcore','protopunk','screamo','psychobilly']):
        target = 'punk'

    elif any(i for i in genres if i in ['metal','death','thrash','metalcore','heavy',
                                        'gothic','deathcore','deathrock','grindcore']):
        target = 'metal'

    elif any(i for i in genres if i in ['soul']):
        target = 'soul'

    elif any(i for i in genres if i in ['jazz','bebop','bop']):
        target = 'jazz'

    elif any(i for i in genres if i in ['country','bluegrass','americana','heartland']):
        target = 'country'

    elif any(i for i in genres if i in ['blues','zydeco']):
        target = 'blues'

    elif any(i for i in genres if i in ['alternative','shoegaze','grunge','indie','garage']):
        target = 'alternative' 

    elif any(i for i in genres if i in ['rock']):
        target = 'rock'

    elif any(i for i in genres if i in ['world','celtic','punjabi','chinese','brazil',
                                        'brazilian','greek','french','angeles','ethnic',
                                        'africa','african','swedish','german','persian',
                                        'iran','bossa','nova']):
        target = 'world'

    else:
        target = 'other'

    return target