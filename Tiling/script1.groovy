// script1.groovy

def imageName = getProjectEntry().getImageName()
def fileName = imageName + '.annotations.json'
def annotations = getAnnotationObjects()
boolean prettyPrint = true
def gson = GsonTools.getInstance(prettyPrint)
def output = gson.toJson(annotations)
new File(fileName).text = output
