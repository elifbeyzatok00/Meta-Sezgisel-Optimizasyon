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
 
Output:
-

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

Output:
-
      
- [ ]  Yapay zeka alg ve nasıl çalıştıklarına detaylıca bak. Özellikle k-nn, arama ağacına ve sezgisel alg. (Binary) bak
