## Modelowanie procesów wydobycia ropy naftowej z użyciem Physics-Informed Neural Networks (PINN)
### Modeling oil extraction using Physics-Informed Neural Networks (PINN)

Kod źródłowy pracy dyplomowej pisanej na Wydziale Informatyki (Informatyka - Data Science) na uczelni AGH w Krakowie.

Celem pracy jest opracowanie i implementacja algorytmu opartego na Physics-Informed Neural Networks (PINN) do symulacji procesów zachodzących w złożach ropy naftowej w ośrodku porowatym.


Struktura repozytorium:

```text
data_SPE/                               # analiza danych przepuszczalności ośrodka porowatego
FEniCS/solver-validation-newton-method  # kod z wykorzystaniem narzędzia FEniCS w celu uzyskania danych do walidacji modeli PINN
maczuga_pinn/                           # Vanilla PINN na podstawie kodu https://github.com/pmaczuga/pinn-notebooks/blob/master/PINN_heat_transfer_2d.ipynb
pinn_hp/                                # PINN na podstawie repozytorium https://github.com/JanTry/PINN_HP
vpinn/                                  # VPINN 
```

