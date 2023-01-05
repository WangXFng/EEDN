# EEDN

## Datasets
<table>
	<tr> <td> Dataset</td> <td> #Users</td> <td> #Items</td> </tr>
	<tr> <td> Douban-book</td> <td> 12,859</td> <td> 22,294</td> </tr>
	<tr> <td> Gowalla</td> <td> 18,737</td> <td> 32,510</td> </tr>
	<tr> <td> Foursquare</td> <td> 7,642</td> <td> 28,483</td> </tr>
	<tr> <td> Yelp challenge round 7</td> <td> 30,887</td> <td> 18,995</td> </tr>
	<tr> <td> Yelp2018</td> <td> 31,668</td> <td> 38,048</td> </tr>
</table>


## Results

<table>
	<tr><th></th><th></th><th>CIKM'18</th><th>SIGIR'20</th><th>SIGIR'21</th><th>APIN'22</th><th>KDD'22</th><th>WWW'22</th><th>SIGIR'22</th><th></th><th></th></tr>
	<tr><th>Dataset</th><th>Metric</th><th>SAE-NAD</th><th>SGL</th><th>LightGCN</th><th>STaTRL</th><th>DirectAU</th><th>NCL</th><th>SIMGCL</th><th>Ours</th><th>Improv.</th></tr>
    <tr><td rowspan="6">Douban-book</td><td>Recall@5 </td><td>0.0661</td><td>0.0640</td><td>0.0708</td><td>0.0693</td><td>0.0700</td><td>0.0753</td><td>0.0795</td><td><b>0.0997</b> </td><td>+34.2%</td></tr>
    <tr><td>NDCG@5 </td><td> 0.1094</td><td>0.1170</td><td>0.1334</td><td>0.1370</td><td>0.1153</td><td>0.1328 </td><td>0.1459</td><td><b>0.1931</b> </td><td>+32.4%</td></tr>
    <tr><td>Recall@10 </td><td>0.0721</td><td>0.0972</td><td>0.1034</td><td>0.0994</td><td>0.0996</td><td>0.1133 </td><td>0.1167</td><td><b>0.1413</b> </td><td>+21.1%</td></tr>
    <tr><td>NDCG@10 </td><td>0.0913</td><td>0.1165</td><td>0.1291</td><td>0.1286</td><td>0.1131</td><td>0.1323</td><td>0.1424</td><td><b>0.1822</b> </td><td>+27.9%</td></tr>
    <tr><td>Recall@20 </td><td>0.1104</td><td>0.1455</td><td>0.1478</td><td>0.1401</td><td>0.1358</td><td>0.1631</td><td>0.1770</td><td><b>0.1917</b> </td><td>+8.3%</td></tr>
    <tr><td>NDCG@20 </td><td>0.0738</td><td>0.1253</td><td>0.1351</td><td>0.1313</td><td>0.1193</td><td>0.1413</td><td>0.1582</td><td><b>0.1840</b> </td><td>+16.3%</td></tr>
    <tr><td rowspan="6">Gowalla</td><td>Recall@5 </td><td>0.0470</td><td>0.0508</td><td>0.0524</td><td>0.0517</td><td>0.0457</td><td>0.0535</td><td>0.0541</td><td><b>0.0602</b> </td><td>+11.3%</td></tr>
    <tr><td>NDCG@5 </td><td>0.0699</td><td>0.0846</td><td>0.0882</td><td>0.0840</td><td>0.0733</td><td>0.0890</td><td>0.0885</td><td><b>0.0996</b> </td><td>+11.9%</td></tr>
    <tr><td>Recall@10 </td><td>0.0731</td><td>0.0781</td><td>0.0824</td><td>0.0803</td><td>0.0729</td><td>0.0849</td><td>0.0835</td><td><b>0.0947</b> </td><td>+11.5%</td></tr>
    <tr><td>NDCG@10 </td><td>0.0596</td><td>0.0875</td><td>0.0919</td><td>0.0876</td><td>0.0781</td><td>0.0938</td><td>0.0924</td><td><b>0.1040</b> </td><td>+10.9%</td></tr>
    <tr><td>Recall@20 </td><td>0.1120</td><td>0.1196</td><td>0.1255</td><td>0.1230</td><td>0.1132</td><td>0.1304</td><td>0.1298</td><td><b>0.1423</b> </td><td>+9.1%</td></tr>
    <tr><td>NDCG@20 </td><td>0.0495</td><td>0.1021</td><td>0.1068</td><td>0.1018</td><td>0.0928</td><td>0.1099</td><td>0.1090</td><td><b>0.1203</b> </td><td>+9.5%</td></tr>
    <tr><td rowspan="6">Foursquare</td><td>Recall@5 </td><td>0.0372</td><td>0.0480</td><td>0.0471</td><td>0.0472</td><td>0.0482</td><td>0.0511</td><td>0.0464</td><td><b>0.0554</b> </td><td>+8.4%</td></tr>
    <tr><td>NDCG@5 </td><td>0.0603</td><td>0.0784</td><td>0.0753</td><td>0.0763</td><td>0.0649</td><td>0.0834 </td><td>0.0725</td><td><b>0.0867</b> </td><td>+4.0%</td></tr>
    <tr><td>Recall@10 </td><td>0.0557</td><td>0.0725</td><td>0.0728</td><td>0.0735</td><td>0.0714</td><td>0.0788 </td><td>0.0732</td><td><b>0.0843</b> </td><td>+7.0%</td></tr>
    <tr><td>NDCG@10 </td><td>0.0496</td><td>0.0795</td><td>0.0772</td><td>0.0778</td><td>0.0696</td><td>0.0854</td><td>0.0764</td><td><b>0.0882</b> </td><td>+3.3%</td></tr>
    <tr><td>Recall@20 </td><td>0.0798</td><td>0.1094</td><td>0.1102</td><td>0.1106</td><td>0.1091</td><td>0.1206</td><td>0.1146</td><td><b>0.1268</b> </td><td>+5.1%</td></tr>
    <tr><td>NDCG@20 </td><td>0.0396</td><td>0.0934</td><td>0.0914</td><td>0.0916</td><td>0.0848</td><td>0.1012</td><td>0.0923</td><td><b>0.1043</b> </td><td>+3.1%</td></tr>
    <tr><td rowspan="6">Yelp challenge round 7</td><td>Recall@5 </td><td>0.0241</td><td>0.0318</td><td>0.0363</td><td>0.0342</td><td>0.0364</td><td>0.0372</td><td>0.0373</td><td><b>0.0386 </b></td><td>+3.5%</td></tr>
    <tr><td>NDCG@5 </td><td>0.0332</td><td>0.0385</td><td>0.0441</td><td>0.0420</td><td>0.0441</td><td>0.0457 </td><td>0.0460</td><td><b>0.0476</b> </td><td>+3.5%</td></tr>
    <tr><td>Recall@10 </td><td>0.0403</td><td>0.0533</td><td>0.0611</td><td>0.0565</td><td>0.0612</td><td>0.0616 </td><td>0.0623</td><td><b>0.0637</b> </td><td>+2.2%</td></tr>
    <tr><td>NDCG@10 </td><td>0.0301</td><td>0.0464</td><td>0.0534</td><td>0.0502</td><td>0.0531</td><td>0.0544</td><td>0.0549</td><td><b>0.0563</b> </td><td>+2.6%</td></tr>
    <tr><td>Recall@20 </td><td>0.0666</td><td>0.0880</td><td>0.0998</td><td>0.0897</td><td>0.0986</td><td>0.0991</td><td>0.1012</td><td><b>0.1020</b> </td><td>+0.8%</td></tr>
    <tr><td>NDCG@20 </td><td>0.0289</td><td>0.0591</td><td>0.0676</td><td>0.0620</td><td>0.0669</td><td>0.0679</td><td>0.0690</td><td><b>0.0702</b> </td><td>+1.7%</td></tr>
    <tr><td rowspan="6">Yelp2018</td><td>Recall@5 </td><td>0.0190</td><td>0.0191</td><td>0.0220</td><td>0.0215</td><td>0.0238</td><td>0.0223</td><td>0.0242</td><td><b>0.0267</b> </td><td>+10.3%</td></tr>
    <tr><td>NDCG@5 </td><td>0.0341</td><td>0.0371</td><td>0.0421</td><td>0.0415</td><td>0.0458</td><td>0.0428</td><td>0.0461</td><td><b>0.0508</b> </td><td>+10.2%</td></tr>
    <tr><td>Recall@10 </td><td>0.0333</td><td>0.0341</td><td>0.0384</td><td>0.0372</td><td>0.0413</td><td>0.0390 </td><td>0.0426</td><td><b>0.0457</b> </td><td>+7.3%</td></tr>
    <tr><td>NDCG@10 </td><td>0.0311</td><td>0.0397</td><td>0.0447</td><td>0.0420</td><td>0.0482</td><td>0.0456</td><td>0.0493</td><td><b>0.0531</b> </td><td>+7.7%</td></tr>
    <tr><td>Recall@20 </td><td>0.0562</td><td>0.0582</td><td>0.0659</td><td>0.0662</td><td>0.0705</td><td>0.0665</td><td>0.0721</td><td><b>0.0760</b> </td><td>+5.4%</td></tr>
    <tr><td>NDCG@20 </td><td>0.0277</td><td>0.0484</td><td>0.0549</td><td>0.0501</td><td>0.0589</td><td>0.0558</td><td>0.0601</td><td><b>0.0639</b> </td><td>+6.3%</td></tr>
</table>
