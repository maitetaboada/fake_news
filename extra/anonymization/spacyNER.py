import spacy
import timeit
import numpy as np





def test_anonymization():
    my_string = "?[Ahmad Khan Rahami] should go to jail and besomebody girlfriend.? ? Rock Legend Gene Simmons commenting on recent NY/NJ bomber Ahmad Khan Rahami Hall of Fame rock legend Gene Simmons made a name for himself playing the bass guitar and singing for the pop-rock band KISS. In more recent yearshe been a reality TV star, a cultural icon, and an outspoken defender of the free market system and personal liberty. The combination of these things has made Simmons marketing gold among more conservative NASCAR loving, flag waving Americans. The folks at TMZ recently ran into Simmons in the airport security line and asked him for his thoughts on the recent bombings. Simmons told TMZ that he thought the Muslim terrorist bomber, Rahami, ?should go to jail and besomebody girlfriend.? Ouch. TMZ then asked Simmons how he would solve the problem of Islamic terrorism in America, to which Simmons replied, rather politically incorrectly, that we needed to do more profiling. ?Nobody says the word ?profile.? Everyone needs to get smart.I ok with profiling. I want you to stop me first. I want you to stop me because I look a certain way. And ifI clear, then profile someone else.It emergency powers during war.There a war going on.? But what about the people who were being unfairlyprofiled, TMZ wondered. Simmons held nothing back in his perfect response. ?It is unfair.That why the words ?too bad? exists.? BOOM. Thank you for being awesome, Mr. Simmons. Maybe when Donald Trump gets elected Presidenthe consider making Gene Simmons the director of Homeland Security? Ooh, and then KISS could play at all of the White House State functions! Hey, I can dream,can I? The views expressed in this opinion article are solely those of their author and are not necessarily either shared or endorsed by EagleRising.com"

    doc = nlp(my_string)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    for i in range(1,100):
        s = "UPDATE: January 18, 2017  Eric Braverman appears to have surfaced. UPDATE: January 10, 2017  The question remains, where is Eric Braverman? As I covered 11 days ago, below, no one has been able to get in contact with the former Clinton Foundation CEO since October 22nd. Thats when Wikileaks released an email in which John Podesta named him as a possible mole within the Clinton Foundation. The mainstream media continues to ignore the story. And is he really missing? I just spoke with his father, Stanley Braverman, who told me, Its nonsense, maam, its not true, before hanging up on me. Note how easy it would be for his dad to disprove it. Instead, he just denied it and hung up. Snopes reports, New Haven, New York City and Washington D.C. police have no record of a missing person with his name, nor is Braverman listed on NamUs, a national database of missing persons maintained by the Justice Department. Emails from The Stream to Braverman, Yale University, where he teaches one course, the FBI, and his husband Neil Brown received no substantive response. The Free Thought Project had no luck getting information from the university. In Custody? Laying Low? Seeking Asylum? Braverman had been hired by Chelsea Clinton to clean up corruption within the foundation, but was forced out by longtime Clinton insiders. The right-wing blogosphere has speculated that the FBI has him in protective FBI custody to protect him from the Clintons or that hes lying low at the Clintons request, in order to avoid being subpoenaed to testify against the organization. Others speculate that he has sought asylum in the Russian embassy. While the rightwing blogosphere is boiling over with theories, the mainstream media isnt covering the story. While the rightwing blogosphere is boiling over with theories, the mainstream media isnt covering the story. CNN is running around the clock coverage of the congressional investigation into Russia allegedly hacking into Podestas emails and the DNCs emails, but has not mentioned the possibility Braverman might have leaked Podestas emails to Wikileaks. Wikileaks founder Julian Assange said a disgruntled Democratic insider gave him the emails, not the Russians. Braverman isnt the only one at the Clinton Foundation who Podesta thought could be a mole. In an email exchange with an ally, Podesta said, could be doug or ira, referring to Doug Bales and Ira Magaziner. Still Missing Since my previous article came out, the only mention of Bravermans disappearance by the mainstream media has been Politicos derisive reference to the story as fake news. At least Snopes revised their entry on the story, changing the status from fake to unknown, with a link to my article. The date to watch is Thursday, January 19th. Thats the Yale class Braverman teaches, Innovation in Government and Society, begins. Well know whether he is missing or not depending on whether he shows up to teach. But dont expect to see a New York Times journalist there. Theyre too busy covering the congressional hearings about the claim that the Russians hacked the emails and gave them to Wikileaks. December 29, 2016  Eric Braverman, the Clinton Foundation CEO from 2013 until 2015, has apparently been missing since October. His absence has fueled speculations in the blogosphere but so far has been ignored by the media. Some speculate, with good reason, that Braverman may have gone into hiding after an email mentioning his name was released by Wikileaks on October 22 of this year. In the March 2015 email exchange, Center for American Progress President Neera Tanden told Clinton campaign manager and confidant John Podesta there was a mole within the Clinton Foundation. Podesta in his reply told Tanden the mole was Braverman. Braverman had abruptly resigned from the Clinton Foundation shortly before this email exchange took place. And then, after the email exchange was made public by Wikileaks, Braverman vanished from the public eye. This seems like a story that someone might want to report. The last evidence of Bravermans public activity was October 12, when he posted his last tweet on Twitter. (Usually he tweets about once a month. His husband, Neil Brown, hasnt tweeted since August, although he rarely tweets.) I left a voicemail on Bravermans personal phone and sent him an email, but received no response. He is still listed as a lecturer at Yale University and, contrary to some reports, there is a record of his lectures going back several years. I contacted the press office and Bravermans department at Yale and received no response. Braverman, the Podesta Leaks and the Clinton Foundation Craig Murray, a former British ambassador to Uzbekistan and a close associate of WikiLeaks founder Julian Assange, told The Daily Mail that Podestas emails were leaked to the organization by a disgruntled insider, not the Russians. Consequently, there are suspicions it may have been Braverman. (Though some of the Podesta emails are dated after Bravermans tenure with the foundation, if he had Podestas password, he could still have accessed his email after leaving.) Re: Tweet from @JoeNBC From:ntanden@americanprogress.org To: john.podesta@gmail.com Date: 2015-03-08 19:48 Subject: Re: Tweet from @JoeNBC Holy Moses. Sent from my iPhone> On Mar 8, 2015, at 5:23 PM, John Podesta <john.podesta@gmail.com> wrote: > > Eric Braverman > > JP > Sent from my iPad > john.podesta@gmail.com > For scheduling: eryn.sepp@gmail.com > >> On Mar 8, 2015, at 4:49 PM, Neera Tanden <ntanden@americanprogress.org> wrote: >> >> @JoeNBC: A source close to the Clintons tell @ron_fournier to follow the money and find the real HRC scandal. http://t.co/lPTQY0L0o4 >> >> Im hoping someone is keeping tabs on Doug Band. Quote in here is from someone who worked in Clinton Foundation. Politico ran a long story about Bravermans ouster in 2015. Based on email correspondence released by Wikileaks, Braverman was apparently hired by Chelsea Clinton to clean up the corruption in the foundation, but then forced out of the foundation by longtime Clinton loyalists; sources say Podesta made him a target. In 2011, Podestas leaked emails show that Chelsea was aggressively calling for an internal investigation. For example, former President Bill Clinton had raised over $1 billion though the foundation to rebuild 100 villages in India, but only $53 million was spent on the project. Also, Braverman resigned at the time Hillary was arranging one of her notorious pay to play deals with foreign leaders: a $12 million contribution from the king of Morocco in exchange for giving a speech. #Wheres Eric: Did Braverman Request Asylum From the Russians? So far Bravermans apparent disappearance has only been discussed by bloggers and fringe websites, which often mix the fact that he has gone silent with other unconfirmed claims. For instance, the site WhatDoesItMean.com reported that Braverman requested asylum in Russia on October 23. The information apparently came from a Russian blogger, who reported it in a rambling blog post on LiveLeak. Thats pretty thin evidence. Moreover, WhatDoesItMean.com is known for posting questionable news stories. The left-leaning, myth-debunking site Snopes labeled the news sites account of Braverman as false, but based its judgment on the fact that website publishes false stories. Aside from this circular argument, Snopes provides no independent evidence for its judgment. I contacted the Russian embassy and received no response. So at the moment, the claim that Braverman requested asylum from the Russians is an unconfirmed rumor. #WheresEric: Could Braverman be in FBI Protective Custody? There are also rumors that Braverman is in FBI protective custody, perhaps in exchange for testifying against the Clintons. Sources within the FBI have said it is likely there will be indictments handed down over the Clinton Foundations pay-to-play schemes. Senior FBI officials told CNN that the investigation into the Clinton Foundation had never ended and is still ongoing. According to The Daily Caller News Foundation, that probe now involves as many as five FBI bureaus across the country: New York, Little Rock, Washington, D.C., Los Angeles and Miami. Would the FBI have hustled Braverman to safety once it was known Podesta had pegged him as a mole? This theory seems plausible, given what we know. But there is still no independent evidence of it. All Questions, No Answers If Braverman is in hiding to protect his life, is it because he leaked the Podesta emails to Wikileaks, and/or is he preparing to testify against the Clintons? The one thing we know is that Braverman has disappeared from the public eye, and that neither he, his husband Neil Brown, nor his family, nor his Yale employers, has made a single public statement to dispel the speculations. The hashtag #WheresEric has been started on Twitter. Follow Rachel on Twitter at Rach_IC."
        doc = nlp(s)
        #replacement = "XXX"
        #replacement_len = len(replacement)
        offset = 0
        for ent in doc.ents:
            s = s[:ent.start_char + offset] + ent.label_ + s[ent.end_char + offset:]
            offset = offset - len(ent.text) + len(ent.label_)
        print(s)



def anonymize(data_file):
    texts = np.load(data_file)
    result = []
    i = 0
    for s in texts:
        if i > 10:
            break
        i = i + 1
        doc = nlp(str(s))
        offset = 0
        for ent in doc.ents:
            s = s[:ent.start_char + offset] + ent.label_ + s[ent.end_char + offset:]
            offset = offset - len(ent.text) + len(ent.label_)
            result.append(s)
        #print(s)
    print(texts[0:10])




def data_anonymization():
    #anonymize test/train/validation files currently in the dump folder
    texts_train = anonymize("../../dump/trainRaw")
    texts_train.dump("../../dump/trainAnon")
    '''
    print("1 Done")
    anonymize("../dump/valid")
    print("2 Done")
    anonymize("../dump/test")
    print("3 Done")
    '''



tic = timeit.default_timer()
#nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_lg')

#test_anonymization()
data_anonymization()

toc = timeit.default_timer()
print(toc - tic)
