import numpy as np
import pandas as pd
import os

from langchain_core.output_parsers import StrOutputParser

# Configuracion splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuracion LLM - GPT
from langchain_openai import ChatOpenAI

from langchain.docstore.document import Document

# Configura tu API key de OpenAI directamente
OPENAI_API_KEY = "tu-api-key-aqui"  # Reemplaza con tu API key real
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4.1")  # o el modelo que prefieras usar

#query
llm.invoke("¿Donde se va a transmitir WWE RAW en 2025?").content

# fuente: https://www.infobae.com/que-puedo-ver/2024/01/23/wwe-en-netflix-raw-y-mas-eventos-llegaran-desde-2025/
noticia_1 = """
Netflix es la nueva casa de la WWE: “Raw” y más eventos especiales se podrán ver en la plataforma

El famoso programa de lucha libre llegará al streaming. WrestleMania, SummerSlam y Royal Rumble también se sumarán el próximo año.

Netflix ha anunciado un histórico acuerdo con la empresa de entretenimiento deportivo WWE. A partir de 2025, la “N” roja será el hogar exclusivo del popular programa Monday Night Raw en Estados Unidos, Canadá, Reino Unido y Latinoamérica.
Este acuerdo, que tiene planes de expandirse a más territorios alrededor de 190 países, representa un cambio trascendental para Raw, ya que se traslada de la televisión tradicional a la plataforma de streaming por primera vez desde su lanzamiento en 1993.

El acuerdo, con una duración de diez años y un valor que supera los 5 mil millones de dólares, también incluirá todos los programas y especiales de la WWE, como SmackDown y NXT, además de los eventos premium en vivo del calibre de WrestleMania, SummerSlam y Royal Rumble.
El gigante del streaming también ofrecerá documentales galardonados de la marca, series originales y proyectos futuros a su audiencia internacional a partir del próximo año.

El impacto de la WWE
Actualmente, Raw es el programa número uno en la cadena USA Network, atrayendo a 17.5 millones de espectadores únicos a lo largo del año, y se mantiene como uno de los programas más exitosos en el demográfico publicitario de 18 a 49 años.
Adicionalmente, WWE cuenta con más de mil millones de seguidores en sus plataformas de redes sociales.

El programa Monday Night Raw es uno de los más icónicos del entretenimiento deportivo, acumulando 1600 episodios desde su debut. Combinando lo mejor del contenido guionizado con el entretenimiento en vivo, el show semanal ha impulsado las carreras de múltiples celebridades en el mundo de la lucha libre con el paso del tiempo, al punto de convertirse en un referente mundial.
"""

# fuente: https://www.emol.com/noticias/Economia/2023/09/28/1108504/luksin-renuncia-presidencia-quinenco-ccu.html
noticia_2 = """
Andrónico Luksic renuncia a la presidencia de Quiñenco y a directorios de otras empresas.

El empresario Andrónico Luksic anunció su renuncia a la presidencia de Quiñenco y de los directorios de otras compañías de las que forma parte. La renuncia se hará efectiva desde el 29 de septiembre. También dejará presidencias de Compañía Cervecerías Unidas (CCU) y LQ Inversiones Financieras (LQIF), las vicepresidencias de Banco de Chile y Compañía Sud Americana de Vapores (CSAV), y el directorio de Invexans.

"Tras un profundo proceso de reflexión he llegado a la convicción de que es momento de alejarme del día a día, de dar paso a otros liderazgos y de permitir que sea el gran equipo de profesionales que hemos construido a lo largo de los años, el que conduzca a nuestras empresas hacia el futuro", sostuvo el empresario.

En Quiñenco, Luksic recalcó haber podido "profundizar la estrategia de diversificación internacional que nos trazamos, al punto que en 2022 más de 90% de la utilidad que obtuvimos provino del extranjero.

También ha sido fundamental la creación, desde 2014, de áreas como relaciones laborales, desarrollo organizacional, asuntos corporativos y sustentabilidad, que incorporaron a nuestra matriz industrial y financiera variables propias de las nuevas exigencias que la sociedad y el mundo demandan de las empresas".

Por su parte, Francisco Pérez Mackenna, quien asumirá la presidencia de Quiñenco, señaló que "la decisión que conocimos hoy es totalmente personal, y fue adoptada por él en forma muy responsable y reflexiva, buscando siempre un mejor futuro para las compañías".

De acuerdo a la determinación adoptada en cada uno de los respectivos directorios, los cambios a concretarse a contar del 29 de diciembre próximo son:
- En el directorio de Quiñenco, asumirá como presidente Pablo Granifo Lavín, en tanto que Paola Luksic Fontbona, actual asesora, se incorporará como directora.
- En Banco de Chile, Francisco Pérez Mackenna será el nuevo vicepresidente, y se sumará en calidad de director Patricio Jottar Nasrallah.
- En CCU, el presidente será Francisco Pérez, y Óscar Hasbún Martínez se incorporará como director.
- En LQIF, Francisco Pérez será el presidente, y el cargo de director será asumido por Rodrigo Hinzpeter Kirberg.
- En Invexans, Vicente Mobarec Katunaric será el nuevo integrante del directorio.
- Y en CSAV, Pablo Granifo se incorpora como director y vicepresidente de la compañía.
"""

doc_1 =  Document(noticia_1)
doc_2 =  Document(noticia_2)
docs =  [doc_1, doc_2]

text_splitter = RecursiveCharacterTextSplitter(
                  chunk_size=450,
                  chunk_overlap=0)

splits = text_splitter.split_documents(docs)

for index, split in enumerate(splits):
  print(f"SPLIT {index + 1}")
  print(split.page_content)
  print("--")

  # Creamos un dataFrame con los textos

chunks = []
for split in splits:
  chunks.append(split.page_content)

df = pd.DataFrame(chunks, columns = ['Text'])

# visualiza los 10 primeros pedacitos
df