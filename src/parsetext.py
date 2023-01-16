#import os
#curdir = os.getcwd()
#filename = os.path.join(curdir, 'example.txt')

# with codecs.open('unicode.rst', encoding='utf-8') as f:
#     for line in f:
#         print repr(line)

with open("./src/example.txt", 'r+',encoding='utf-8') as file, open("./src/accounts.txt", "w",encoding='utf-8') as f3:
    for line in file:
        
        finaloutput = line
        # print(line.rstrip())
        idx = finaloutput.find("font-size")
        while(idx != -1):

            ##read
            afterstyletext = finaloutput[idx - 1:]
            beforestyletext = finaloutput[:idx - 1]
            startafterstyle = afterstyletext[1:].find("\"")
            end = afterstyletext.find("p")
            start = afterstyletext.find(":")
            theintger = int(afterstyletext[start+1: end])
            ##replace
            string = "{{fontSize:" + str(theintger * 1.335) + "}}"
            #idx = findstyle.find("style=")
            finaloutput = beforestyletext + string + afterstyletext[startafterstyle + 2:]
            idx = finaloutput.find("font-size")
            #print("first")

            
        f3.write(finaloutput)
        #print("eol")