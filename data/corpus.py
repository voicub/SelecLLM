"""data/corpus.py — synthetic multilingual corpus with known labels."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import random
from dataclasses import dataclass
from typing import List, Optional
from config import CFG

random.seed(CFG.random_seed)

_T = {
 "en":{"positive":["This is absolutely wonderful and I am very happy with the result.",
                   "The conference was fantastic and I learned so much today.",
                   "Great job everyone, the project is going really well.",
                   "I love the new features and the performance is excellent.",
                   "We achieved outstanding results and the team deserves recognition."],
       "neutral": ["The meeting starts at three o'clock in the afternoon.",
                   "Please send the document to the address listed below.",
                   "The system processes approximately five hundred requests per second.",
                   "We need to review the technical specifications before the deadline.",
                   "The report has been submitted and is currently under review."],
       "negative":["This situation is very frustrating and I am not satisfied at all.",
                   "The network connection keeps failing and the latency is terrible.",
                   "I am disappointed with the results and we need to fix this urgently.",
                   "The errors keep occurring and nobody seems to know the cause.",
                   "This is unacceptable and we must address the problem immediately."]},
 "de":{"positive":["Das ist wirklich ausgezeichnet und ich bin sehr zufrieden damit.",
                   "Die Konferenz war fantastisch und sehr lehrreich fuer alle.",
                   "Tolle Arbeit, das Projekt entwickelt sich sehr gut.",
                   "Ich freue mich sehr ueber die neuen Funktionen des Systems.",
                   "Wir haben hervorragende Ergebnisse erzielt."],
       "neutral": ["Das Treffen beginnt um drei Uhr nachmittags im Konferenzraum.",
                   "Bitte senden Sie das Dokument an die angegebene Adresse.",
                   "Das System verarbeitet ungefaehr fuenfhundert Anfragen pro Sekunde.",
                   "Wir muessen die technischen Spezifikationen vor dem Termin pruefen.",
                   "Der Bericht wurde eingereicht und wird derzeit geprueft."],
       "negative":["Diese Situation ist sehr frustrierend und ich bin ueberhaupt nicht zufrieden.",
                   "Die Netzwerkverbindung bricht immer wieder ab und die Latenz ist schrecklich.",
                   "Ich bin enttaeuscht von den Ergebnissen und wir muessen das dringend beheben.",
                   "Die Fehler treten immer wieder auf und niemand kennt die Ursache.",
                   "Das ist inakzeptabel und wir muessen das Problem sofort angehen."]},
 "fr":{"positive":["C'est absolument merveilleux et je suis tres satisfait du resultat.",
                   "La conference etait fantastique et j'ai beaucoup appris aujourd'hui.",
                   "Excellent travail a tous, le projet se deroule vraiment bien.",
                   "J'adore les nouvelles fonctionnalites et les performances sont excellentes.",
                   "Nous avons obtenu des resultats remarquables."],
       "neutral": ["La reunion commence a quinze heures dans la salle de conference.",
                   "Veuillez envoyer le document a l'adresse indiquee ci-dessous.",
                   "Le systeme traite environ cinq cents requetes par seconde.",
                   "Nous devons examiner les specifications techniques avant la date limite.",
                   "Le rapport a ete soumis et est actuellement en cours d'examen."],
       "negative":["Cette situation est tres frustrante et je ne suis pas du tout satisfait.",
                   "La connexion reseau continue de tomber et la latence est terrible.",
                   "Je suis decu des resultats et nous devons resoudre cela d'urgence.",
                   "Les erreurs continuent de se produire et personne ne semble connaitre la cause.",
                   "C'est inacceptable et nous devons regler ce probleme immediatement."]},
 "es":{"positive":["Esto es absolutamente maravilloso y estoy muy contento con el resultado.",
                   "La conferencia fue fantastica y aprendi muchisimo hoy.",
                   "Excelente trabajo a todos, el proyecto va muy bien.",
                   "Me encantan las nuevas funciones y el rendimiento es excelente.",
                   "Hemos logrado resultados sobresalientes."],
       "neutral": ["La reunion comienza a las tres de la tarde en la sala de conferencias.",
                   "Por favor envie el documento a la direccion indicada abajo.",
                   "El sistema procesa aproximadamente quinientas solicitudes por segundo.",
                   "Necesitamos revisar las especificaciones tecnicas antes del plazo.",
                   "El informe ha sido enviado y actualmente esta siendo revisado."],
       "negative":["Esta situacion es muy frustrante y no estoy satisfecho en absoluto.",
                   "La conexion de red sigue fallando y la latencia es terrible.",
                   "Estoy decepcionado con los resultados y necesitamos arreglar esto urgentemente.",
                   "Los errores siguen ocurriendo y nadie parece conocer la causa.",
                   "Esto es inaceptable y debemos abordar el problema de inmediato."]},
 "it":{"positive":["Questo e assolutamente meraviglioso e sono molto soddisfatto del risultato.",
                   "La conferenza e stata fantastica e ho imparato moltissimo oggi.",
                   "Ottimo lavoro a tutti, il progetto sta andando molto bene.",
                   "Adoro le nuove funzionalita e le prestazioni sono eccellenti.",
                   "Abbiamo ottenuto risultati notevoli."],
       "neutral": ["La riunione inizia alle tre del pomeriggio nella sala conferenze.",
                   "Si prega di inviare il documento all'indirizzo indicato di seguito.",
                   "Il sistema elabora circa cinquecento richieste al secondo.",
                   "Dobbiamo esaminare le specifiche tecniche prima della scadenza.",
                   "Il rapporto e stato presentato e attualmente e in fase di revisione."],
       "negative":["Questa situazione e molto frustrante e non sono affatto soddisfatto.",
                   "La connessione di rete continua a cadere e la latenza e terribile.",
                   "Sono deluso dai risultati e dobbiamo risolvere urgentemente questo problema.",
                   "Gli errori continuano a verificarsi e nessuno sembra conoscere la causa.",
                   "Questo e inaccettabile e dobbiamo affrontare il problema immediatamente."]},
 "nl":{"positive":["Dit is absoluut geweldig en ik ben erg blij met het resultaat.",
                   "De conferentie was fantastisch en ik heb vandaag veel geleerd.",
                   "Goed werk iedereen, het project gaat echt heel goed."],
       "neutral": ["De vergadering begint om drie uur in de middag in de vergaderzaal.",
                   "Stuur het document alstublieft naar het hieronder vermelde adres.",
                   "Het systeem verwerkt ongeveer vijfhonderd verzoeken per seconde."],
       "negative":["Deze situatie is erg frustrerend en ik ben helemaal niet tevreden.",
                   "De netwerkverbinding blijft uitvallen en de latentie is verschrikkelijk.",
                   "Ik ben teleurgesteld in de resultaten en we moeten dit dringend oplossen."]},
 "pt":{"positive":["Isso e absolutamente maravilhoso e estou muito satisfeito com o resultado.",
                   "A conferencia foi fantastica e aprendi muito hoje.",
                   "Otimo trabalho a todos, o projeto esta indo muito bem."],
       "neutral": ["A reuniao comeca as tres horas da tarde na sala de conferencias.",
                   "Por favor envie o documento para o endereco indicado abaixo.",
                   "O sistema processa aproximadamente quinhentas solicitacoes por segundo."],
       "negative":["Esta situacao e muito frustrante e nao estou satisfeito de forma alguma.",
                   "A conexao de rede continua falhando e a latencia e terrivel.",
                   "Estou decepcionado com os resultados e precisamos corrigir isso urgentemente."]},
 "ro":{"positive":["Acesta este absolut minunat si sunt foarte multumit de rezultat.",
                   "Conferinta a fost fantastica si am invatat foarte mult astazi.",
                   "Munca excelenta tuturor, proiectul merge foarte bine."],
       "neutral": ["Sedinta incepe la ora trei dupa-amiaza in sala de conferinte.",
                   "Va rugam sa trimiteti documentul la adresa indicata mai jos.",
                   "Sistemul proceseaza aproximativ cinci sute de cereri pe secunda."],
       "negative":["Aceasta situatie este foarte frustranta si nu sunt deloc multumit.",
                   "Conexiunea la retea continua sa se defecteze si latenta este teribile.",
                   "Sunt dezamagit de rezultate si trebuie sa rezolvam urgent aceasta problema."]},
 "ru":{"positive":["Eto absoliutno zamechatielno, i ia ochen dovolen rezultatom.",
                   "Konferientsiya byla fantasticheskoy, i ya mnogo uznal segodnya.",
                   "Otlichnaya rabota vsem, proekt idyot ochen khorosho."],
       "neutral": ["Soveshchaniye nachinayetsya v tri chasa dnya v konferents-zale.",
                   "Pozhaluysta, otpravte dokument po ukazannomu nizhe adresu.",
                   "Sistema obrabativayet okolo pyatisot zaprosov v sekundu."],
       "negative":["Eta situatsiya ochen rasstraivayet, i ya sovsem ne dovolen.",
                   "Setevoye soyedineniye prodolzhayet obrivatsya, a zaderzhka uzhasna.",
                   "Ya razocharovan rezultatami i nam nuzhno srochn ispravit eto."]},
 "zh":{"positive":["Zhe zhen shi tai bang le, wo dui jie guo fei chang man yi.",
                   "Hui yi fei chang jing cai, wo jin tian xue dao le hen duo dong xi.",
                   "Da jia gan de hen hao, xiang mu jin zhan fei chang shun li."],
       "neutral": ["Hui yi xia wu san dian zai hui yi shi kai shi.",
                   "Qing ba wen jian fa song dao xia fang suo zhi de di zhi.",
                   "Xi tong mei miao chu li da yue wu bai ge qing qiu."],
       "negative":["Zhe zhong qing kuang fei chang ling ren ju sang, wo yi dian ye bu man yi.",
                   "Wang luo lian jie yi zhi zhong duan, yan chi fei chang ke pa.",
                   "Wo dui jie guo gan dao shi wang, wo men xu yao jin ji xiu fu ci wen ti."]},
}

SENTIMENTS = ["positive","neutral","negative"]

@dataclass
class Utterance:
    text: str; lang: str; sentiment: str; speaker_id: int; utterance_id: int

def generate_corpus(n_per_lang_sentiment: int = 20,
                    languages: Optional[List[str]] = None) -> List[Utterance]:
    if languages is None: languages = CFG.adv_languages
    corpus, uid = [], 0
    for lang in languages:
        tmpl = _T.get(lang, _T["en"])
        for sent in SENTIMENTS:
            pool = tmpl.get(sent, tmpl["neutral"])
            for i in range(n_per_lang_sentiment):
                text = pool[i % len(pool)]
                if i >= len(pool):
                    text += " " + random.choice(["Indeed.","Furthermore.","Additionally.",
                                                  "However.","Moreover.","Therefore."])
                corpus.append(Utterance(text=text, lang=lang, sentiment=sent,
                                        speaker_id=random.randint(0,4),
                                        utterance_id=uid)); uid += 1
    random.shuffle(corpus); return corpus
