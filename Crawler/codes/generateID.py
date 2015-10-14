import re
class ID_Generator:

    # read xml file and save the first #num to a ID file
    # category is for the type of xml file - string
    def saveID(self,path,category,num):
        re_qid = re.compile(r'(?<=Id=").*(?=" PostTypeId)') 
        re_uid = re.compile(r'(?<=Id=").*(?=" Reputation)') 
        data_file = open(path,'r')
        save_file = open("id.txt","w")
        count = 1
        while(1):
            if count >num:
                break
            line = data_file.readline()
            Id = re_uid.findall(line)
            for _Id in Id:
                save_file.write(category+str(count)+"\t"+_Id+"\n")
                count += 1
                print Id



_ID =  ID_Generator()
_ID.saveID("/Users/admin/CMU/codes/data/webStructure/stackoverflow/android.stackexchange.com/Posts.xml","question",300)
#_ID.saveID("/Users/admin/CMU/codes/data/webStructure/stackoverflow/android.stackexchange.com/Users.xml","user",200)
