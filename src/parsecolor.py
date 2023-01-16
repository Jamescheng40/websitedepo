with open("./src/accounts.txt", 'r+',encoding='utf-8') as file, open("./src/colors.txt", "w",encoding='utf-8') as f3:
    for line in file:
        
        finaloutput = line
        # print(line.rstrip())
        idx = finaloutput.find("\"color")
        while(idx != -1):

            ##read
            afterstyletext = finaloutput[idx+1:]
            beforestyletext = finaloutput[:idx]
            startafterstyle = afterstyletext[1:].find("\"")
            end = afterstyletext.find("\"")
            start = afterstyletext.find(":")
            theintger = afterstyletext[start+1: end]
            ##replace
            string = "{{color:" + "\"" + theintger + "\"" + "}}"
            #idx = findstyle.find("style=")
            finaloutput = beforestyletext + string + afterstyletext[startafterstyle + 2:]
            idx = finaloutput.find("\"color")
            #print("first")

            
        f3.write(finaloutput)
        #print("eol")