// ----------- IMPORT compressed JSON

import java.util.zip.GZIPInputStream
InputStream fileStream = new FileInputStream("d:/1L1_nuclei_anno.json.gz");
InputStream gzipStream = new GZIPInputStream(fileStream);
Reader decoder = new InputStreamReader(gzipStream, 'ascii');
BufferedReader buffered = new BufferedReader(decoder);

def gson = GsonTools.getInstance(true)

// Read the annotations
def type = new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>() {}.getType()
def deserializedAnnotations = gson.fromJson(buffered, type)

// Set the annotations to have a different name (so we can identify them) & add to the current image
// deserializedAnnotations.eachWithIndex {annotation, i -> annotation.setName('New annotation ' + (i+1))}   # --- THIS WON"T WORK IN CURRENT VERSION
addObjects(deserializedAnnotations)

