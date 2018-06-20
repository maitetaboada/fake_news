import spacy
import timeit



nlp = spacy.load('en_core_web_sm')
#s = "Apple is looking at buying U.K. startup for $1 billion. Obama thinks that it's not a great idea because Clinton won the election."

my_string = "?[Ahmad Khan Rahami] should go to jail and besomebody girlfriend.? ? Rock Legend Gene Simmons commenting on recent NY/NJ bomber Ahmad Khan Rahami Hall of Fame rock legend Gene Simmons made a name for himself playing the bass guitar and singing for the pop-rock band KISS. In more recent yearshe been a reality TV star, a cultural icon, and an outspoken defender of the free market system and personal liberty. The combination of these things has made Simmons marketing gold among more conservative NASCAR loving, flag waving Americans. The folks at TMZ recently ran into Simmons in the airport security line and asked him for his thoughts on the recent bombings. Simmons told TMZ that he thought the Muslim terrorist bomber, Rahami, ?should go to jail and besomebody girlfriend.? Ouch. TMZ then asked Simmons how he would solve the problem of Islamic terrorism in America, to which Simmons replied, rather politically incorrectly, that we needed to do more profiling. ?Nobody says the word ?profile.? Everyone needs to get smart.I ok with profiling. I want you to stop me first. I want you to stop me because I look a certain way. And ifI clear, then profile someone else.It emergency powers during war.There a war going on.? But what about the people who were being unfairlyprofiled, TMZ wondered. Simmons held nothing back in his perfect response. ?It is unfair.That why the words ?too bad? exists.? BOOM. Thank you for being awesome, Mr. Simmons. Maybe when Donald Trump gets elected Presidenthe consider making Gene Simmons the director of Homeland Security? Ooh, and then KISS could play at all of the White House State functions! Hey, I can dream,can I? The views expressed in this opinion article are solely those of their author and are not necessarily either shared or endorsed by EagleRising.com"
doc = nlp(my_string)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

tic = timeit.default_timer()
for i in range(1,1000):
    s = "?[Ahmad Khan Rahami] should go to jail and besomebody girlfriend.? ? Rock Legend Gene Simmons commenting on recent NY/NJ bomber Ahmad Khan Rahami Hall of Fame rock legend Gene Simmons made a name for himself playing the bass guitar and singing for the pop-rock band KISS. In more recent yearshe been a reality TV star, a cultural icon, and an outspoken defender of the free market system and personal liberty. The combination of these things has made Simmons marketing gold among more conservative NASCAR loving, flag waving Americans. The folks at TMZ recently ran into Simmons in the airport security line and asked him for his thoughts on the recent bombings. Simmons told TMZ that he thought the Muslim terrorist bomber, Rahami, ?should go to jail and besomebody girlfriend.? Ouch. TMZ then asked Simmons how he would solve the problem of Islamic terrorism in America, to which Simmons replied, rather politically incorrectly, that we needed to do more profiling. ?Nobody says the word ?profile.? Everyone needs to get smart.I ok with profiling. I want you to stop me first. I want you to stop me because I look a certain way. And ifI clear, then profile someone else.It emergency powers during war.There a war going on.? But what about the people who were being unfairlyprofiled, TMZ wondered. Simmons held nothing back in his perfect response. ?It is unfair.That why the words ?too bad? exists.? BOOM. Thank you for being awesome, Mr. Simmons. Maybe when Donald Trump gets elected Presidenthe consider making Gene Simmons the director of Homeland Security? Ooh, and then KISS could play at all of the White House State functions! Hey, I can dream,can I? The views expressed in this opinion article are solely those of their author and are not necessarily either shared or endorsed by EagleRising.com"
    doc = nlp(s)
    replacement = "XXX"
    replacement_len = len(replacement)
    offset = 0
    for ent in doc.ents:
        s = s[:ent.start_char + offset] + replacement + s[ent.end_char + offset:]
        offset = offset - len(ent.text) + replacement_len
    #print(s)
toc = timeit.default_timer()
#print(s)
print(toc - tic)

