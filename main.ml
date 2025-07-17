open Data
open Nn

let write_metrics_csv filename metrics =
  let lines =
    "epoch,train_loss,test_loss,train_acc,test_acc\n"
    :: List.map (fun (epoch, train_loss, test_loss, train_acc, test_acc) ->
         Printf.sprintf "%d,%.6f,%.6f,%.6f,%.6f"
           epoch train_loss test_loss train_acc test_acc
       ) metrics
  in
  let content = String.concat "\n" lines ^ "\n" in
  let oc = open_out filename in
  output_string oc content;
  close_out oc


let classification_report params testset =
  let tp = ref 0 in
  let tn = ref 0 in
  let fp = ref 0 in
  let fn = ref 0 in
  List.iter (fun (x, y) ->
    let (_, _, _, a4, _) = forward_pass params x in
    let y_pred = if List.hd a4 >= 0.5 then 1.0 else 0.0 in
    match (y, y_pred) with
    | (1.0, 1.0) -> incr tp
    | (0.0, 0.0) -> incr tn
    | (0.0, 1.0) -> incr fp
    | (1.0, 0.0) -> incr fn
    | _ -> ()
  ) testset;
  let tp = float_of_int !tp in
  let tn = float_of_int !tn in
  let fp = float_of_int !fp in
  let fn = float_of_int !fn in
  let precision = if tp +. fp = 0. then 0. else tp /. (tp +. fp) in
  let recall = if tp +. fn = 0. then 0. else tp /. (tp +. fn) in
  let f1 = if precision +. recall = 0. then 0. else 2. *. precision *. recall /. (precision +. recall) in
  let support_pos = tp +. fn in
  let support_neg = tn +. fp in
  Printf.printf "\nClassification report (test set):\n";
  Printf.printf "%-10s %-10s %-10s %-10s\n" "Class" "Precision" "Recall" "F1-score";
  Printf.printf "%-10s %-10.4f %-10.4f %-10.4f (support: %.0f)\n" "1" precision recall f1 support_pos;
  let precision0 = if tn +. fn = 0. then 0. else tn /. (tn +. fn) in
  let recall0 = if tn +. fp = 0. then 0. else tn /. (tn +. fp) in
  let f10 = if precision0 +. recall0 = 0. then 0. else 2. *. precision0 *. recall0 /. (precision0 +. recall0) in
  Printf.printf "%-10s %-10.4f %-10.4f %-10.4f (support: %.0f)\n" "0" precision0 recall0 f10 support_neg;
  Printf.printf "\n";
  ()


let write_classification_report_csv filename params testset =
  let tp = ref 0 in
  let tn = ref 0 in
  let fp = ref 0 in
  let fn = ref 0 in
  List.iter (fun (x, y) ->
    let (_, _, _, a4, _) = forward_pass params x in
    let y_pred = if List.hd a4 >= 0.5 then 1.0 else 0.0 in
    match (y, y_pred) with
    | (1.0, 1.0) -> incr tp
    | (0.0, 0.0) -> incr tn
    | (0.0, 1.0) -> incr fp
    | (1.0, 0.0) -> incr fn
    | _ -> ()
  ) testset;
  let tp = float_of_int !tp in
  let tn = float_of_int !tn in
  let fp = float_of_int !fp in
  let fn = float_of_int !fn in
  let precision1 = if tp +. fp = 0. then 0. else tp /. (tp +. fp) in
  let recall1 = if tp +. fn = 0. then 0. else tp /. (tp +. fn) in
  let f1_1 = if precision1 +. recall1 = 0. then 0. else 2. *. precision1 *. recall1 /. (precision1 +. recall1) in
  let support1 = tp +. fn in
  let precision0 = if tn +. fn = 0. then 0. else tn /. (tn +. fn) in
  let recall0 = if tn +. fp = 0. then 0. else tn /. (tn +. fp) in
  let f1_0 = if precision0 +. recall0 = 0. then 0. else 2. *. precision0 *. recall0 /. (precision0 +. recall0) in
  let support0 = tn +. fp in
  let oc = open_out filename in
  output_string oc "Class,Precision,Recall,F1-score,Support\n";
  Printf.fprintf oc "1,%.4f,%.4f,%.4f,%.0f\n" precision1 recall1 f1_1 support1;
  Printf.fprintf oc "0,%.4f,%.4f,%.4f,%.0f\n" precision0 recall0 f1_0 support0;
  close_out oc


let () =
  Random.init 42;
  let diabetes_file = "diabetes.csv" in
  let trainset, testset = load_and_split_diabetes_csv diabetes_file in

  let params = init_params () in
  let epochs = 100 in
  let lr = 0.05 in
  let batch_size = 48 in
  let patience =20 in
  let trained_params, metrics = train ~learning_rate:lr ~patience:patience ~batch_size:batch_size params trainset testset epochs in

  write_metrics_csv "metrics.csv" metrics;
  write_classification_report_csv "classification_report.csv" trained_params testset;

  let final_train_acc = List.nth metrics (List.length metrics - 1) |> (fun (_,_,_,train_acc,_) -> train_acc) in
  let final_test_acc = List.nth metrics (List.length metrics - 1) |> (fun (_,_,_,_,test_acc) -> test_acc) in
  Printf.printf "Trained on %d samples for %d epochs.\n" (List.length trainset) epochs;
  Printf.printf "Train accuracy: %.2f%%\n" (100. *. final_train_acc);
  Printf.printf "Test accuracy: %.2f%%\n" (100. *. final_test_acc);

  classification_report trained_params testset; 