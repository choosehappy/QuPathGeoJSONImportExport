// --- remove measurements, not needed but makes file smaller
Set annotationMeasurements = []

getDetectionObjects().each{it.getMeasurementList().getMeasurementNames().each{annotationMeasurements << it}}
//println(annotationMeasurements)

annotationMeasurements.each{ removeMeasurements(qupath.lib.objects.PathCellObject, it);}


// write to file
boolean prettyPrint = false // false results in smaller file sizes and thus faster loading times, at the cost of nice formating
def gson = GsonTools.getInstance(prettyPrint)
def annotations = getDetectionObjects()
//println gson.toJson(annotations) // you can check here but this will be HUGE and take a long time to parse


// automatic output filename, otherwise set explicitly
String imageLocation = getCurrentImageData().getServer().getPath()
outfname = imageLocation.split("file:/")[1]+".json"



File file = new File(outfname)
file.withWriter('UTF-8') {
    gson.toJson(annotations,it)
}
