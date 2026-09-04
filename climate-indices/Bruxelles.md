## Température à 2 mètres

```{=html}
<p align="center">
```
`<img src="./figures/T2M_Bruxelles_last365d.png" width="1200">`{=html}
```{=html}
</p>
```
*Pour la même figure sur les années précédentes, voir
[ici](./T2MAllYears_Bruxelles)*

`<br>`{=html} `<br>`{=html}

```{=html}
<p align="center">
```
`<img src="./figures/T2M_MinMax_Bruxelles_last365d.png" width="1200">`{=html}
```{=html}
</p>
```
## Géopotentiel à 500 hPa

```{=html}
<p align="center">
```
`<img src="./figures/Z500_Bruxelles_last365d.png" width="1200">`{=html}
```{=html}
</p>
```
`<br>`{=html} `<br>`{=html}

```{=html}
<p align="center">
```
`<img src="./figures/Z500_MinMax_Bruxelles_last365d.png" width="1200">`{=html}
```{=html}
</p>
```
`<br>`{=html} `<br>`{=html}

```{=html}
<p align="center">
```
`<img src="./figures/climatology_Z500_Bruxelles.png" width="1200">`{=html}
```{=html}
</p>
```
## Bienvenue

Cette page reprend des statistiques issues de la **[réanalyse
atmosphérique
ERA5](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803)**
(Hersbach et al., 2020) au-dessus de Bruxelles, en quasi temps réel
(délai de quelques jours).

Deux variables sont présentées :

-   la **température de l'air à 2 mètres**, qui permet notamment
    d'apprécier la variabilité saisonnière de la température, les vagues
    de chaleur ou de froid et les tendances multi-décennales ;
-   la **hauteur géopotentielle à 500 hPa**, qui décrit la circulation
    atmosphérique dans la moyenne troposphère. Les valeurs présentées
    correspondent au niveau de pression 500 hPa et sont exprimées en
    mètres géopotentiels.

Ces données sont accessibles publiquement via le [*Climate Data
Store*](https://cds.climate.copernicus.eu/) du programme Copernicus
d'observation de la Terre de l'Union Européenne.

**Clause de non-responsabilité** : ces données, issues d'un modèle
numérique atmosphérique contraint par des observations, n'ont pas pour
vocation de remplacer des données issues de stations météorologiques.
Elles permettent en revanche de caractériser de manière cohérente
l'évolution des conditions atmosphériques sur plusieurs décennies.

Cette page est actualisée une fois par jour, automatiquement.

Les données brutes servant à produire les figures ci-dessus sont
téléchargeables ci-dessous, sous la forme de fichiers CSV lisibles par
des logiciels de tableur classiques.

## Données brutes et statistiques

### Température à 2 mètres

**[Données à l'échelle horaire (Clic droit +
Enregistrer)](./output/hourly_T2M_Bruxelles.csv.gz)**

**[Données agrégées à l'échelle journalière (moyenne, minimum, maximum)
(Clic droit +
Enregistrer)](./output/dailyStatistics_T2m_Bruxelles.csv)**

Le script (Python3) qui produit ces données est disponible
[ici](./era5_temperature.py).

### Géopotentiel à 500 hPa

**[Données à l'échelle 6-horaire (Clic droit +
Enregistrer)](./output/six_hourly_Z500_Bruxelles.csv.gz)**

**[Données agrégées à l'échelle journalière (moyenne, minimum, maximum)
(Clic droit +
Enregistrer)](./output/daily_statistics_Z500_Bruxelles.csv)**

Le script (Python3) qui produit ces données est disponible
[ici](./era5_geopotential_500hPa.py).

## Autres lieux

(A continuer)

## Références

Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A.,
Muñoz‐Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D.,
Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold,
P., Biavati, G., Bidlot, J., Bonavita, M., ... Thépaut, J.-N. (2020).
The ERA5 Global Reanalysis. Quarterly Journal of the Royal
Meteorological Society, https://doi.org/10.1002/qj.3803

## Contact

Pour toute question ou suggestion d'amélioration, contacter **[François
Massonnet](mailto:francois.massonnet@uclouvain.be)**
