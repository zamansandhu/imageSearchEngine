import xml.etree.ElementTree as et
doc = et.parse('Bird_1_rucod.xml')
nodes = doc.findall('.//Row')
ResultSet_Py_List = []
for node in nodes:
    inner = []
    for child in node:
        inner.append(child.text)
    ResultSet_Py_List.append(inner)
ResultSet_Py_List=[[child.text for child in node] for node in nodes]
"""from django.core.management import call_command

call_command('process_xslt',
             'C:\Users\Muhammad Usman\PycharmProjects\ImgSearchEngine\Imgsearchengine\DATASET_FINAL_WORK\01 Bird',
             'transform.xslt', '--save')
<mapping xsl:version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <model model="import_data.Imagestore">
        <xsl:for-each select="//div[@class='question-summary narrow']">
            <item key="">
                <field name="title">
                    <xsl:value-of select=".//a[@class='question-hyperlink']"/>
                </field>
                <xsl:for-each select=".//a[@class='post-tag']">
                    <m2mk model="import_data.Tag">
                        <xsl:attribute name="key">
                            <xsl:value-of select="generate-id(.)"/>
                        </xsl:attribute>
                    </m2mk>
                </xsl:for-each>
            </item>
        </xsl:for-each>
    </model>
</mapping>

import lucene
lucene.initVM(lucene.CLASSPATH)
writer = lucene.IndexWriter("/home/lucene/index", lucene.StandardAnalyzer())
searcher = lucene.IndexSearcher("/home/lucene/index")
doc = lucene.Document()
doc.add(lucene.Field(
  "title",
  "This is a long title for an essay",
  lucene.Field.Store.YES,
  lucene.Field.Index.TOKENIZED))
writer.addDocument(doc)
writer.optimize()
writer.close()
parser = lucene.QueryParser("title", lucene.StandardAnalyzer())
query = parser.parse("+happy movie year:1990")
hits = searcher.search(query)
for h in hits:
  hit = lucene.Hit.cast_(h)
  id, doc = hit.getId(), hit.getDocument()
for f in doc.getFields():
  field = lucene.Field.cast_(f)
  (k, v) = field.name(), field.stringValue()
  print("field name =", "its value = %s" % (k, v))

"""
