def gson = GsonTools.getInstance(true)
def json = new File("Path to *_output_annotations.json").text
//println json

// Read the annotations
def type = new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>() {}.getType()
def deserializedAnnotations = gson.fromJson(json, type)

// Set the annotations to have a different name (so we can identify them) & add to the current image
addObjects(deserializedAnnotations)   
