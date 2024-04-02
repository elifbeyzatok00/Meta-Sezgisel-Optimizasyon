# Meta-Sezgisel-Optimizasyon (Meta-Heuristic-Optimization)


➡️Kodları çalıştırmak için online aşağıdaki online compilerları kullanabilirsiniz:
* [Online Matlab](https://matlab.mathworks.com/)
* [Online Python Compiler](https://www.programiz.com/python-programming/online-compiler/)


### Homerworks


- [ ]  Gezgin satıcı (traveling salesman problem) problemi matlab’da çözümle genetik alg. vb. İle kodla
      
### Solution1:

Gezgin satıcı problemi, bir satıcının bir dizi şehri (müşteriyi) ziyaret etmesi gereken ve her şehri sadece bir kez ziyaret edebileceği bir optimizasyon problemidir. Amacımız, belirli bir şehir dizisini (rotayı) seçerek, toplam yolculuk maliyetini minimize etmektir.

Genetik algoritma, bu tür problemleri çözmek için yaygın olarak kullanılan bir meta-sezgisel optimizasyon algoritmasıdır. MATLAB'da gezgin satıcı problemi için genetik algoritma kullanarak bir çözüm yazalım:

```matlab
% Şehirlerin koordinatlarını oluştur
nCities = 10; % Şehir sayısı
cityCoordinates = 100*rand(nCities, 2); % 0 ile 100 arasında rastgele koordinatlar

% Genetik Algoritma Parametreleri
populationSize = 50;
numGenerations = 100;
mutationRate = 0.01;

% Başlangıç popülasyonunu oluştur
population = zeros(populationSize, nCities);
for i = 1:populationSize
    population(i, :) = randperm(nCities);
end

% Başlangıç en iyi rotayı ve maliyetini tut
bestRoute = [];
bestCost = Inf;

% Genetik algoritmayı çalıştır
for generation = 1:numGenerations
    % Her bireyin uygunluğunu hesapla
    costs = zeros(1, populationSize);
    for i = 1:populationSize
        costs(i) = calculateCost(population(i, :), cityCoordinates);
    end
    
    % En iyi rotayı ve maliyeti bul
    [minCost, minIndex] = min(costs);
    if minCost < bestCost
        bestCost = minCost;
        bestRoute = population(minIndex, :);
    end
    
    % Yeni nesil oluştur
    newPopulation = zeros(populationSize, nCities);
    for i = 1:2:populationSize
        % Ebeveynleri seç
        parents = selection(population, costs);
        % Çaprazlama yap
        children = crossover(parents);
        % Mutasyon
        children = mutation(children, mutationRate);
        % Yeni nesile ekle
        newPopulation(i, :) = children(1, :);
        newPopulation(i+1, :) = children(2, :);
    end
    
    % Yeni nesli eski nesil olarak güncelle
    population = newPopulation;
end

% En iyi rotayı ekrana yazdır
disp('En iyi rotayı bulundu:');
disp(bestRoute);
disp(['Maliyet: ', num2str(bestCost)]);

% Helper Fonksiyonlar

function cost = calculateCost(route, coordinates)
    nCities = length(route);
    cost = 0;
    for i = 1:nCities-1
        cost = cost + norm(coordinates(route(i), :) - coordinates(route(i+1), :));
    end
    cost = cost + norm(coordinates(route(nCities), :) - coordinates(route(1), :));
end

function parents = selection(population, costs)
    [~, sortedIndices] = sort(costs);
    % Rulet tekerleği seçimi
    probabilities = 1./(1 + costs(sortedIndices));
    probabilities = probabilities / sum(probabilities);
    cumulativeProb = cumsum(probabilities);
    parents = zeros(2, size(population, 2));
    for i = 1:2
        r = rand();
        idx = find(cumulativeProb >= r, 1, 'first');
        parents(i, :) = population(sortedIndices(idx), :);
    end
end

function children = crossover(parents)
    nCities = size(parents, 2);
    children = zeros(2, nCities);
    % Tek nokta çaprazlama
    crossoverPoint = randi([1, nCities-1]);
    children(1, 1:crossoverPoint) = parents(1, 1:crossoverPoint);
    [~, IA, ~] = intersect(parents(2, :), children(1, 1:crossoverPoint), 'stable');
    children(1, crossoverPoint+1:end) = setdiff(parents(2, :), children(1, 1:crossoverPoint), 'stable');
    children(2, 1:crossoverPoint) = parents(2, 1:crossoverPoint);
    [~, IA, ~] = intersect(parents(1, :), children(2, 1:crossoverPoint), 'stable');
    children(2, crossoverPoint+1:end) = setdiff(parents(1, :), children(2, 1:crossoverPoint), 'stable');
end

function mutatedChildren = mutation(children, mutationRate)
    mutatedChildren = children;
    nCities = size(children, 2);
    for i = 1:size(children, 1)
        if rand() < mutationRate
            % Swap mutation
            idx = randperm(nCities, 2);
            mutatedChildren(i, [idx(1), idx(2)]) = mutatedChildren(i, [idx(2), idx(1)]);
        end
    end
end
```

Bu kod, belirli sayıda şehir ve koordinatlarla başlayarak genetik algoritmayı kullanarak gezgin satıcı problemi çözer. Kodun çalışması için MATLAB ortamında çalıştırılması gerekmektedir. Bu kodda, genetik algoritmanın her bir neslinde, seçim, çaprazlama ve mutasyon adımları gerçekleştirilir. Bu adımlar genetik algoritmanın temel bileşenleridir ve gezgin satıcı problemi için uygulanırlar.

Output:
![image](https://github.com/elifbeyzatok00/Meta-Sezgisel-Optimizasyon/assets/102792446/9e8150b8-b0ef-466d-8e9f-4a3ceca9618e)

### Solution2:
## Gezgin Satıcı Problemini MATLAB'da Çözümleme

Gezgin satıcı problemi (GSP), bir satıcının bir dizi şehri ziyaret etmesi ve her şehre yalnızca bir kez uğraması gereken en kısa rotayı bulma problemidir. Bu problem, optimizasyon problemleri için klasik bir örnektir ve birçok farklı alanda uygulamaya sahiptir.

**MATLAB'da GSP'yi çözmek için:**

1. **Şehirlerin konumlarını temsil eden bir matris oluşturun.** Bu matris, her şehir arasındaki mesafeyi veya koordinatları içerebilir.

2. **Genetik algoritma (GA) gibi bir optimizasyon algoritması seçin.** GA'lar, bir çözüm popülasyonunu evrimleştirerek çalışır. Her çözüm, şehirlerin bir sırasını temsil eder.

3. **GA'nın parametrelerini ayarlayın.** Bu parametreler, popülasyon büyüklüğü, mutasyon oranı ve çaprazlama oranını içerir.

4. **GA'yı çalıştırın.** GA, en iyi çözümü bulana kadar bir dizi yineleme gerçekleştirecektir.

5. **En iyi çözümü analiz edin.** En iyi çözüm, en kısa rotayı temsil eden şehirlerin sırası olacaktır.

**MATLAB'da GSP'yi çözmek için aşağıdaki kodları kullanabilirsiniz:**

```matlab
% Şehirlerin konumlarını temsil eden matris
distances = [
    0, 10, 20, 30, 40,
    10, 0, 15, 25, 35,
    20, 15, 0, 10, 20,
    30, 25, 10, 0, 10,
    40, 35, 20, 10, 0
];

% GA parametreleri
populationSize = 100;
mutationRate = 0.1;
crossoverRate = 0.5;

% GA'yı çalıştır
[bestSolution, bestDistance] = ga(distances, populationSize, mutationRate, crossoverRate);

% En iyi çözümü yazdır
disp(['En iyi rota:', num2str(bestSolution)]);
disp(['En kısa mesafe:', num2str(bestDistance)]);
```

**Kodun Açıklaması:**

* `distances` matrisi, şehirlerin arasındaki mesafeleri içerir.
* `populationSize` değişkeni, popülasyondaki çözüm sayısını belirtir.
* `mutationRate` ve `crossoverRate` değişkenleri, GA'nın mutasyon ve çaprazlama operatörlerini ne sıklıkla kullanacağını belirtir.
* `ga` fonksiyonu, GA'yı çalıştırır ve en iyi çözümü ve en kısa mesafeyi döndürür.
* `disp` fonksiyonları, en iyi çözümü ve en kısa mesafeyi yazdırır.

**Notlar:**

* Bu kod, GSP'yi çözmek için basit bir GA kullanır. Daha karmaşık GA'lar daha iyi sonuçlar verebilir.
* Bu kod, yalnızca küçük problem boyutları için çalışabilir. Daha büyük problemler için daha karmaşık algoritmalar kullanmak gerekir.

**Gezgin satıcı problemini çözmek için MATLAB'da kullanabileceğiniz diğer algoritmalar:**

* Bencil Genetik Algoritmalar
* Tabu Arama
* Simüle edilmiş Tavlama
* Karınca Kolonisi Optimizasyonu

**Daha fazla bilgi için:**

* Gezgin Satıcı Problemi (Wikipedia): [https://tr.wikipedia.org/wiki/Gezgin_sat%C4%B1c%C4%B1_problemi]
* Genetik Algoritmalar (Wikipedia): [https://tr.wikipedia.org/wiki/Genetik_algoritma](https://tr.wikipedia.org/wiki/Genetik_algoritma)
* MATLAB Optimizasyon Araç Kutusu: [https://www.mathworks.com/products/optimization.html](https://www.mathworks.com/products/optimization.html)

Output:
![image](https://github.com/elifbeyzatok00/Meta-Sezgisel-Optimizasyon/assets/102792446/fd60ce0e-45ef-40ed-b4ea-15f3091b0976)


- [ ]  Binary kodlama + sezgisel alg ile future selection yani özellik seçimi gerçekleştir
      
Özellik seçimi (feature selection) işlemi, bir veri kümesindeki en önemli özellikleri belirleyerek gereksiz özelliklerden kurtulmayı amaçlar. Bu, modelin daha iyi performans göstermesini sağlar, aynı zamanda hesaplama yükünü azaltır ve modelin genelleme yeteneğini artırır. Binary kodlama ve sezgisel algoritmalar, özellik seçimi için kullanılabilecek yöntemlerden bazılarıdır. İşte bu iki yöntemin birleştirilerek özellik seçimi yapılmasını sağlayacak bir yaklaşım:

1. **Veri Hazırlığı**: İlk adım, veri kümesini temizlemek ve uygun formata getirmektir. Eksik verileri doldurmak, kategorik verileri sayısal forma dönüştürmek gibi adımlar bu aşamada gerçekleştirilir.

2. **Binary Kodlama**: Kategorik özellikler, modelin işleyebileceği sayısal formata dönüştürülür. Bu adımda, One-Hot Encoding gibi teknikler kullanılabilir.

3. **Sezgisel Algoritma Seçimi**: Özellik seçimi için bir sezgisel algoritma seçmek gerekir. Genetik algoritma, sürü optimizasyonu, parçacık sürüsü optimizasyonu gibi algoritmalar kullanılabilir. Bu algoritmalardan birini seçmek, özelliklerin optimize edilmesine ve en iyi kombinasyonun bulunmasına yardımcı olur.

4. **Özellik Seçimi**: Seçilen sezgisel algoritmayı kullanarak en iyi özellik kombinasyonunu belirle. Bu adımda, algoritmanın özellik seçiminde hangi kriterleri değerlendireceği belirlenir. Örneğin, en yüksek doğruluk sağlayan özellik kombinasyonunu seçebilirsiniz.

5. **Model Eğitimi ve Değerlendirme**: Seçilen özelliklerle bir model oluşturun ve eğitin. Daha sonra, bu modeli doğrulamak için ayrılmış bir test veri kümesi üzerinde değerlendirin. Modelin performansını ölçmek için uygun metrikleri kullanın (doğruluk, hassasiyet, geri çağırma vb.).

6. **Gerekirse İterasyon**: Modelin performansını artırmak için gerekiyorsa adımları tekrarlayın. Farklı sezgisel algoritmaları veya farklı parametre ayarlarını deneyebilirsiniz.

Bu adımları takip ederek, binary kodlama ve sezgisel algoritmaları birleştirerek veri kümesindeki en önemli özellikleri seçebilir ve daha iyi bir model elde edebilirsiniz. Bu süreç, veri setinize ve probleminize bağlı olarak farklılık gösterebilir, bu nedenle deneme yanılma yöntemini kullanarak en iyi sonuçları elde etmek önemlidir.

Örnek_python:
Bu örnek, özellik seçimi için bir genetik algoritma kullanır. Kod, özellik seçimini gerçekleştirmek için scikit-learn kütüphanesini ve genetik algoritma için DEAP kütüphanesini kullanır.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms

# Örnek veri kümesi oluşturma
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Veri kümesini eğitim ve test olarak bölmek
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitness fonksiyonu - doğruluk
def evaluate(individual, X, y):
    selected_features = [bool(i) for i in individual]
    X_selected = X[:, selected_features]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_selected, y)
    y_pred = clf.predict(X_selected)
    return accuracy_score(y, y_pred),

# Genetik algoritma için tanımlamalar
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.choice, [0, 1])
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X[0]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate, X=X_train, y=y_train)

# Genetik algoritma parametreleri
population_size = 50
num_generations = 10

pop = toolbox.population(n=population_size)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, verbose=True)

# En iyi bireyin seçilmesi
best_individual = tools.selBest(pop, k=1)[0]
selected_features = [bool(i) for i in best_individual]

# En iyi özelliklerin kullanılmasıyla modelin eğitilmesi ve test edilmesi
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_selected, y_train)
y_pred = clf.predict(X_test_selected)

# Test verisi üzerinde modelin değerlendirilmesi
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy with selected features:", accuracy)
```

Bu kod, scikit-learn kütüphanesini kullanarak bir sınıflandırma veri seti oluşturur, genetik algoritmayı kullanarak özellik seçimini gerçekleştirir ve ardından seçilen özelliklerle bir RandomForestClassifier modeli eğitir. Son olarak, modelin test veri seti üzerindeki performansını değerlendirir ve doğruluğu yazdırır.

<!--
Output:
-
-->

Örnek_matlab:
Aynı işlemi MATLAB'da da gerçekleştirebiliriz. MATLAB'da genetik algoritma için özel bir araç olan Global Optimization Toolbox'un `ga` fonksiyonunu kullanabiliriz. İşte MATLAB kodu:

```matlab
% Örnek veri kümesi oluşturma
rng(42); % Tekrarlanabilirlik için rastgele tohumu ayarla
X = randn(1000, 20); % 1000 örnek, 20 özellik
y = randi([0 1], 1000, 1); % 0 ve 1 sınıfları arasında rastgele etiketler

% Veri kümesini eğitim ve test olarak bölmek
cv = cvpartition(y, 'Holdout', 0.2);
X_train = X(cv.training,:);
y_train = y(cv.training,:);
X_test = X(cv.test,:);
y_test = y(cv.test,:);

% Fitness fonksiyonu - doğruluk
fitnessFunc = @(individual) evaluate(individual, X_train, y_train);

function accuracy = evaluate(individual, X, y)
    selected_features = logical(individual);
    X_selected = X(:, selected_features);
    mdl = fitensemble(X_selected, y, 'Bag', 100, 'Tree', 'Type', 'Classification');
    y_pred = predict(mdl, X_selected);
    accuracy = sum(y_pred == y) / numel(y);
end

% Genetik algoritma parametreleri
numFeatures = size(X, 2);
opts = gaoptimset('PopulationSize', 50, 'Generations', 10, 'CrossoverFraction', 0.5, 'MutationFcn', {@mutationuniform, 0.05});
[selected_features, ~] = ga(fitnessFunc, numFeatures, opts);

% Seçilen özelliklerin kullanılmasıyla modelin eğitilmesi ve test edilmesi
X_train_selected = X_train(:, selected_features);
X_test_selected = X_test(:, selected_features);
mdl = fitensemble(X_train_selected, y_train, 'Bag', 100, 'Tree', 'Type', 'Classification');
y_pred = predict(mdl, X_test_selected);

% Test verisi üzerinde modelin değerlendirilmesi
accuracy = sum(y_pred == y_test) / numel(y_test);
fprintf('Test accuracy with selected features: %.2f%%\n', accuracy * 100);
```

Bu MATLAB kodu, öncelikle bir veri seti oluşturur, ardından genetik algoritmayı kullanarak özellik seçimini gerçekleştirir ve en iyi özellik kombinasyonunu seçer. Son olarak, seçilen özelliklerle bir sınıflandırma modeli eğitir ve test veri seti üzerinde performansını değerlendirir.

<!--
Output:
-
-->
      
- [ ]  Yapay zeka alg ve nasıl çalıştıklarına detaylıca bak. Özellikle k-nn, arama ağacına ve sezgisel alg. (Binary) bak
K-nn nedir, nasıl çalışır?

K-nearest neighbors (K-NN), bir sınıflandırma veya regresyon probleminde kullanılan basit bir makine öğrenimi algoritmasıdır. Temel prensibi, bir veri noktasını sınıflandırmak veya tahmin etmek için en yakınındaki K sayıda eğitim örneğine bakmaktır. K-NN, genellikle basit ama etkili bir şekilde çalışır ve çeşitli uygulamalarda kullanılabilir.

K-NN nasıl çalışır:
Elbette, işte K-NN algoritmasının daha detaylı adımları:

1. **Veri Kümesinin Hazırlanması:**
   - İlk adım veri kümesinin hazırlanmasıdır. Bu veri kümesi, özellikler ve bu özelliklere karşılık gelen hedef değişkenlerden oluşur. Örneğin, bir ev fiyatı tahmin etmek istiyorsak, özellikler evin metrekareleri, oda sayısı, konumu vb. olabilir.

2. **Normalizasyon veya Standartlaştırma:**
   - Özellikler arasındaki değer aralıkları farklı olabilir. Bu nedenle, normalizasyon veya standartlaştırma gibi işlemlerle özellik değerlerini aynı ölçekte tutmak önemlidir. Bu adım, veri ön işleme sürecinin bir parçasıdır.

3. **K Değerinin Seçilmesi:**
   - K-NN algoritmasında, K parametresi komşu sayısını belirtir. Bu parametrenin doğru bir şekilde seçilmesi önemlidir. Genellikle, K değeri tek sayılar arasından seçilir (çift sayılar için çoğunluk oylamasında bağlanma riskini azaltmak için).

4. **Mesafe Ölçümü:**
   - K-NN'de, genellikle kullanılan mesafe ölçümleri Öklid mesafesi veya Manhattan mesafesi gibi yöntemlerdir. Öklid mesafesi en yaygın olarak kullanılan mesafe ölçüsüdür.

5. **Test Verilerinin Sınıflandırılması veya Tahmini:**
   - Sınıflandırma için: Test verisi noktasının her eğitim veri noktasıyla arasındaki mesafeler hesaplanır ve en yakın K komşular seçilir. Ardından, bu komşuların sınıfları incelenir ve test veri noktası bu komşuların sınıfları arasında çoğunluk oylaması ile sınıflandırılır.
   - Regresyon için: Test veri noktasının her eğitim veri noktasıyla arasındaki mesafeler hesaplanır ve en yakın K komşular seçilir. Ardından, bu komşuların hedef değişken değerlerinin ortalaması alınarak test veri noktasının tahmini değeri hesaplanır.

6. **Sonuçların Değerlendirilmesi:**
   - Algoritmanın performansını değerlendirmek için genellikle doğruluk, hassasiyet, geri çağırma, ortalama kare hata (MSE) gibi metrikler kullanılır. Bu metrikler, algoritmanın ne kadar iyi çalıştığını anlamak için kullanılır.

Bu adımları takip ederek K-NN algoritmasıyla bir sınıflandırma veya regresyon problemi çözebilirsiniz. Ancak, doğru K değeri seçimi, mesafe ölçümü gibi faktörlerin seçimi ve veri kümesinin kalitesi gibi birçok faktör algoritmanın başarısını etkileyebilir.

Arama ağaçları, karar ağaçları, random forest nedir ve aşamaları nelerdir?
Arama ağaçları, karar ağaçları ve Random Forest, farklı kavramlardır ve her biri farklı amaçlar için kullanılan farklı algoritmalardır.

1. **Arama Ağaçları (Search Trees)**:
Arama ağaçları, genellikle veri yapıları olarak kullanılır ve veri elemanlarının hızlı bir şekilde aranmasını sağlar. En popüler örneği ikili arama ağaçlarıdır. İkili arama ağacı, her düğümün en fazla iki çocuğu olan bir ağaç yapısıdır. Temel işlevi, bir elemanın varlığını hızlı bir şekilde kontrol etmek ve sıralı bir şekilde saklanmış veri elemanlarını tutmaktır. Ağaç, genellikle soldan sağa sıralıdır ve herhangi bir düğüm, sol alt ağaçtaki tüm elemanlardan daha büyük ve sağ alt ağaçtaki tüm elemanlardan daha küçüktür. Ayrıca, arama ağaçları, öğe ekleme, arama ve silme gibi temel işlemleri gerçekleştirmek için kullanılır.

2. **Karar Ağaçları (Decision Trees)**:
Karar ağaçları, sınıflandırma ve regresyon problemleri için kullanılan bir makine öğrenimi tekniğidir. Veri kümesinin belirli özelliklerine dayanarak bir hedef değişkeni tahmin etmek için kullanılır. Karar ağaçları, bir kök düğümden başlar ve her iç düğüm, bir özellikle ilişkilendirilmiş bir karar kuralını temsil eder. Her bir dal, bir özellik değerine göre ayrılır ve sonuç olarak bir yaprak düğümünde bir tahmin veya karar verilir. Karar ağaçları, veri setindeki karmaşık ilişkileri modellemek için kullanılabilir ve kolayca yorumlanabilirler.

3. **Random Forest**:
Random Forest, bir ensemble (topluluk) öğrenme algoritmasıdır ve karar ağaçlarının bir araya gelmesiyle oluşur. Her bir karar ağacı rastgele örnekleme ve rastgele özellik seçimi teknikleri kullanılarak eğitilir. Random Forest, sınıflandırma ve regresyon problemleri için kullanılabilir ve genellikle overfitting (aşırı uyum) problemlerini azaltmak için kullanılır. Aşağıda Random Forest'ın ana adımları verilmiştir:

   - Veri örnekleme (Bootstrap sampling)
   - Özellik seçimi (Feature selection)
   - Karar ağaçlarının oluşturulması (Tree building)
   - Tahmin (Prediction)

Bu adımlar, Random Forest'ın her bir karar ağacını eğitme ve sonuçları birleştirme sürecini tanımlar.

Sezgisel Algoritmalar nedir, nasıl çalışır?
Sezgisel algoritmalar, doğal fenomenlerden, insan davranışlarından veya diğer karmaşık sistemlerden esinlenerek tasarlanan ve problem çözme sürecinde sezgi veya deneme-yanılma yöntemlerini kullanan algoritmalardır. Bu algoritmalar, bilgisayar biliminde ve yapay zekâ alanında geniş bir kullanım alanına sahiptirler. Birçok sezgisel algoritma, problemi optimize etmeye veya bir çözüm alanında arama yapmaya odaklanır.

Sezgisel algoritmaların genel çalışma prensibi, doğal süreçlerden veya davranışlardan esinlenerek çözüm alanında potansiyel çözümleri aramaktır. Bu çözümler, genellikle bir uygunluk fonksiyonu tarafından değerlendirilir ve en iyi çözüm, bu fonksiyona göre seçilir. İşte bazı popüler sezgisel algoritma türleri:

1. **Genetik Algoritmalar (Genetic Algorithms - GA)**: Evrimsel teoriden esinlenen bu algoritma, bir popülasyonu çeşitli çözümlerle temsil eder ve doğal seçilim, çaprazlama ve mutasyon operatörleri kullanarak daha iyi çözümleri üretmek için iteratif bir süreç uygular.

2. **Parçacık Sürü Optimizasyonu (Particle Swarm Optimization - PSO)**: Kuş sürüsü veya bal arısı kolonisi gibi doğal organizmaların davranışlarından esinlenen PSO, bir çözüm alanındaki olası çözümleri birer parçacık olarak temsil eder ve bu parçacıklar, en iyi çözümü bulmak için birbirleriyle etkileşirler.

3. **Karınca Kolonisi Optimizasyonu (Ant Colony Optimization - ACO)**: Karıncaların yiyecek kaynaklarını bulmak için kullandığı iz sürme davranışından esinlenen ACO, problem alanında birçok olası çözümü keşfetmek için bir koloni yaklaşımı kullanır ve izler boyunca feromon izlerini güncelleyerek en iyi çözüme ulaşmaya çalışır.

4. **Simüle Edilen Tavlama (Simulated Annealing)**: Metal işleme sanayisindeki ısıtma ve soğutma süreçlerinden esinlenen bu algoritma, olası çözümleri kabul etme olasılığına dayalı olarak hareket eder ve sıcaklık parametresini yavaşça azaltarak çözüm alanında daha iyi bir çözüm arar.

5. **Yapay Arı Kolonisi Algoritması (Artificial Bee Colony Algorithm - ABC)**: Bal arıları tarafından yiyecek kaynaklarını bulmak için kullanılan davranışlardan esinlenen ABC, bir çözüm alanında olası çözümleri keşfetmek için bir arı kolonisi yaklaşımı kullanır.

Bu sezgisel algoritmalar, optimizasyon problemlerini çözmek, veri madenciliği, yapay sinir ağları eğitimi, oyun stratejileri geliştirme ve daha pek çok alanda kullanılabilirler. Her bir algoritmanın kendine özgü avantajları, dezavantajları ve kullanım alanları vardır; bu nedenle, belirli bir probleme en uygun olanı seçmek önemlidir.


Sezgisel Algoritmalar (Binary) nedir, nasıl çalışır?
Sezgisel algoritmalar, problem çözme sürecinde sezgi veya deneme-yanılma yöntemlerini kullanarak çözüm alanında arama yapmak için tasarlanmış bir tür algoritmadır. Bu algoritmalar, birçok farklı alanda kullanılabilir, özellikle karmaşık ve zorlu optimizasyon problemlerini çözmek için yaygın olarak kullanılırlar.

Sezgisel algoritmaların bir türü olan Binary Sezgisel Algoritmalar (BSA), genellikle karmaşık ve çok boyutlu bir arama alanında optimal bir çözümü bulmak için kullanılır. Bu algoritma, bir dizi bit dizgesi kullanarak çözüm alanını temsil eder. Her bir bit, bir çözümün bir parçasını temsil eder.

BSA'nın çalışma prensibi genellikle şu adımları içerir:

1. **Başlangıç Popülasyonunun Oluşturulması**: İlk adımda, rastgele veya belirli bir yöntemle başlangıç popülasyonu oluşturulur. Bu popülasyon, problem alanındaki olası çözümleri temsil eden bit dizgelerinden oluşur.

2. **Uygunluk Fonksiyonunun Hesaplanması**: Her bir çözüm için uygunluk fonksiyonu hesaplanır. Uygunluk fonksiyonu, bir çözümün ne kadar iyi olduğunu ölçen bir metrik veya objektif fonksiyondur. Bu fonksiyon, problemle ilgili özel gereksinimlere bağlı olarak belirlenir.

3. **Seçim**: Seçim operatörü, popülasyon içinden uygunluklarına göre çözümleri seçer. Daha iyi uygunluk değerlerine sahip çözümler, bir sonraki nesle aktarılacak olan bireyler olarak seçilir.

4. **Çaprazlama (Crossover)**: Seçilen çözümler, çaprazlama operatörü kullanılarak yeni çözümler oluşturmak için birleştirilir. Bu adım, genellikle iki veya daha fazla çözümün bit dizgeleri üzerinde gerçekleştirilir. Bu, potansiyel olarak daha iyi çözümlerin keşfedilmesini sağlar.

5. **Mutasyon**: Mutasyon operatörü, popülasyon içindeki çözümlerin çeşitliliğini artırmak için kullanılır. Rastgele seçilen bitlerin değerleri, rastgele olarak değiştirilir. Bu, yeni ve potansiyel olarak daha iyi çözümlerin keşfedilmesine yardımcı olur.

6. **Yeni Popülasyonun Oluşturulması**: Seçim, çaprazlama ve mutasyon adımlarının ardından yeni bir popülasyon oluşturulur. Bu popülasyon, bir sonraki nesil için potansiyel çözümleri temsil eder.

7. **Durma Koşulunun Kontrol Edilmesi**: Belirli bir durma koşulu sağlanıncaya kadar adımlar tekrarlanır. Bu koşul genellikle belirli bir iterasyon sayısına veya bir çözüm kalitesine ulaşıldığında gerçekleşir.

8. **En İyi Çözümün Seçilmesi**: Son olarak, en iyi uygunluğa sahip çözüm, problem için bir yaklaşım olarak seçilir.

BSA, çeşitli optimizasyon problemlerini çözmek için kullanılabilir. Bunlar, genetik algoritmalar (GA), parçacık sürü optimizasyonu (PSO) ve karınca kolonisi optimizasyonu (ACO) gibi diğer sezgisel algoritmalarla karşılaştırılabilir.
