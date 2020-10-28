
// ----------- IMPORT JSON


def gson = GsonTools.getInstance(true)

def json = new File("D:/research/chuv_alex_mel/1l1_2rois_annotated.json").text
println json


// Read the annotations
def type = new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>() {}.getType()
def deserializedAnnotations = gson.fromJson(json, type)

// Set the annotations to have a different name (so we can identify them) & add to the current image
// deserializedAnnotations.eachWithIndex {annotation, i -> annotation.setName('New annotation ' + (i+1))}   # --- THIS WON"T WORK IN CURRENT VERSION
addObjects(deserializedAnnotations)   


