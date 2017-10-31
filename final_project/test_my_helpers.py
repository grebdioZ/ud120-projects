import os
import unittest

from final_project.my_helpers import *


class ParseEmailTest( unittest.TestCase ):
    def test_parseEmail( self ):
        info = parseEmail( "h:/dev/ud120-projects/maildir/brawner-s/inbox/101")
        self.assertEqual( "Mon, 26 Nov 2001 13:46:56 -0800 (PST)", info["Date"])
        self.assertEqual( "scott.neal@enron.com", info["From"])
        self.assertEqual( "f..brawner@enron.com", info["To"])
        self.assertEqual( "FW: BUG Rebooking", info["Subject"])
        self.assertTrue( info["Body"].startswith("\n\n\n -----Original Message-----\nFrom: 	Neal, Scott") )
        self.assertTrue( info["Body"].endswith("Where:	ECS  5102\n") )

    def test_parseHeader(self):
        header = """
Message-ID: <4821308.1075861110477.JavaMail.evans@thyme>
Date: Thu, 14 Mar 2002 14:36:45 -0800 (PST)
From: kharrell@periwinklefoundation.org
To: schick.ana@enron.com, miller <"".jackie@enron.com>, yauch.janet@enron.com, 
	jariel.jenifer@enron.com, mauel.joan@enron.com, simon.judy@enron.com
Subject: Board Retreat
Cc: psorrells@periwinklefoundation.org
Bcc: pressler.townes@enron.com"""
        info = parseHeader(header)
        self.assertEqual("<4821308.1075861110477.JavaMail.evans@thyme>", info["ID"])
        self.assertEqual("Thu, 14 Mar 2002 14:36:45 -0800 (PST)", info["Date"])
        self.assertEqual( "kharrell@periwinklefoundation.org", info["From"])
        self.assertEqual( "schick.ana@enron.com miller.jackie@enron.com yauch.janet@enron.com jariel.jenifer@enron.com mauel.joan@enron.com simon.judy@enron.com", info["To"])
        self.assertEqual( "Board Retreat", info["Subject"])
        self.assertEqual( "psorrells@periwinklefoundation.org", info["Cc"])
        self.assertEqual( "pressler.townes@enron.com", info["Bcc"])

    def test_parseHeaderMissingLines(self):
        header = """
Message-ID: <4821308.1075861110477.JavaMail.evans@thyme>
Date: Thu, 14 Mar 2002 14:36:45 -0800 (PST)
From: kharrell@periwinklefoundation.org
Subject: Board Retreat
Bcc: pressler.townes@enron.com"""
        info = parseHeader(header)
        self.assertEqual("<4821308.1075861110477.JavaMail.evans@thyme>", info["ID"])
        self.assertEqual("Thu, 14 Mar 2002 14:36:45 -0800 (PST)", info["Date"])
        self.assertEqual("kharrell@periwinklefoundation.org", info["From"])
        self.assertTrue("To" not in info)
        self.assertEqual("Board Retreat", info["Subject"])
        self.assertTrue("Cc" not in info)
        self.assertEqual("pressler.townes@enron.com", info["Bcc"])

    def test_getSentEmailFileNames(self):
        mailFiles = getEmailFileNames("a..hughes@enron.com", "from" )
        self.assertTrue(mailFiles)
        for fileName in mailFiles:
            self.assertTrue(os.path.isfile( fileName ), fileName)

    def test_getReceivedEmailFileNames(self):
        mailFiles = getEmailFileNames("a..hughes@enron.com", "to" )
        self.assertTrue(mailFiles)
        for fileName in mailFiles:
            self.assertTrue(os.path.isfile( fileName ), fileName)

    @unittest.skip("Takes too long")
    def test_getEmailFileNamesForPOI(self):
        emailLists = os.listdir(EMAIL_LINK_ROOT)
        cnt = 0
        step = 100
        for emailList in emailLists[::step]:
            cnt += step
            try:
                prefix, address = parseEmailListName(emailList)
            except Exception as e:
                print "Processing", cnt, "of", len(emailLists)
                print("List {} failed processing: '{}'!".format(emailList, e))
                continue
            mailFiles = getEmailFileNames(address, prefix)
            self.assertTrue(mailFiles)
            for fileName in mailFiles:
                # Unfortunately, some of the referenced emails appear to be missing, so we don't assert here, we
                # just notify
                if not os.path.isfile( fileName ):
                    log( "Processing", cnt, "of", len(emailLists) )
                    print("List {} links missing email '{}'!".format(emailList, fileName ))

    def test_getNameForAddressFails(self):
        with open("final_project_dataset.pkl", "r") as data_file:
            data_dict = cPickle.load(data_file)
        self.assertEqual( None, getNameForAddress( data_dict, "scott.neal@enron.com" )  )

    def test_getNameForAddress(self):
        with open("final_project_dataset.pkl", "r") as data_file:
            data_dict = cPickle.load(data_file)
        self.assertEqual( "GIBBS DANA R", getNameForAddress( data_dict, "dana.gibbs@enron.com" )  )