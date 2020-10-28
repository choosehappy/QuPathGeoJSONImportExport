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


File file = new File('d:/1L1_nuclei.json')
file.withWriter('UTF-8') {
    gson.toJson(annotations,it)
}
