# Meta-Sezgisel-Optimizasyon
Meta-Heuristic-Optimization

➡️Kodları çalıştırmak için [Online Matlab](https://matlab.mathworks.com/) kullanabilirsiniz
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
![image](https://github.com/elifbeyzatok00/Meta-Sezgisel-Optimizasyon/assets/102792446/fd60ce0e-45ef-40ed-b4ea-15f3091b0976)

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

- [ ]  Binary kodlama + sezgisel alg ile future selection yani özellik seçimi gerçekleştir
- [ ]  Yapay zeka alg ve nasıl çalıştıklarına detaylıca bak. Özellikle k-nn, arama ağacına ve sezgisel alg. (Binary) bak
