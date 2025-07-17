open Math

let sublist l start len =
  let rec aux i j acc = function
    | [] -> List.rev acc
    | x :: xs when i > 0 -> aux (i - 1) j acc xs
    | x :: xs when j > 0 -> aux 0 (j - 1) (x :: acc) xs
    | _ -> List.rev acc
  in
  aux start len [] l


let median l =
  let sorted = List.sort compare l in
  let n = List.length sorted in
  if n = 0 then 0. else
    if n mod 2 = 1 then List.nth sorted (n/2)
    else (List.nth sorted (n/2 - 1) +. List.nth sorted (n/2)) /. 2.

let quantile l q =
  let sorted = List.sort compare l in
  let idx = int_of_float (q *. float_of_int (List.length l - 1)) in
  List.nth sorted idx

let best_feature_indices = [0; 1; 5; 6] (* Pregnancies, Glucose, BMI, DiabetesPedigreeFunction *)

let select_best_features xs =
  List.map (fun row -> List.map (fun i -> List.nth row i) best_feature_indices) xs

let impute_and_cap xs =
  let cols = List.init 8 (fun i -> List.map (fun row -> List.nth row i) xs) in
  let cols' = List.mapi (fun j col ->
    match j with
    | 0 -> (* Pregnancies: cap outliers (IQR) *)
        let q1 = quantile col 0.25 in
        let q3 = quantile col 0.75 in
        let iqr = q3 -. q1 in
        let lw = q1 -. 1.5 *. iqr in
        let uw = q3 +. 1.5 *. iqr in
        List.map (fun x -> if x < lw then quantile col 0.05 else if x > uw then quantile col 0.95 else x) col
    | 1 -> (* Glucose: 0→median, no capping *)
        let med = median (List.filter ((<>) 0.) col) in
        List.map (fun x -> if x = 0. then med else x) col
    | 2 -> (* BloodPressure: 0→median, cap outliers (IQR) *)
        let med = median (List.filter ((<>) 0.) col) in
        let col = List.map (fun x -> if x = 0. then med else x) col in
        let q1 = quantile col 0.25 in
        let q3 = quantile col 0.75 in
        let iqr = q3 -. q1 in
        let lw = q1 -. 1.5 *. iqr in
        let uw = q3 +. 1.5 *. iqr in
        List.map (fun x -> if x < lw then quantile col 0.05 else if x > uw then quantile col 0.95 else x) col
    | 3 -> (* SkinThickness: 0→median, cap outliers (IQR) *)
        let med = median (List.filter ((<>) 0.) col) in
        let col = List.map (fun x -> if x = 0. then med else x) col in
        let q1 = quantile col 0.25 in
        let q3 = quantile col 0.75 in
        let iqr = q3 -. q1 in
        let lw = q1 -. 1.5 *. iqr in
        let uw = q3 +. 1.5 *. iqr in
        List.map (fun x -> if x < lw then quantile col 0.05 else if x > uw then quantile col 0.95 else x) col
    | 4 -> (* Insulin: 0→q60, cap outliers (IQR) *)
        let q60 = quantile col 0.6 in
        let col = List.map (fun x -> if x = 0. then q60 else x) col in
        let q1 = quantile col 0.25 in
        let q3 = quantile col 0.75 in
        let iqr = q3 -. q1 in
        let lw = q1 -. 1.5 *. iqr in
        let uw = q3 +. 1.5 *. iqr in
        List.map (fun x -> if x < lw then quantile col 0.05 else if x > uw then quantile col 0.95 else x) col
    | 5 -> (* BMI: 0→median, cap outliers (IQR) *)
        let med = median (List.filter ((<>) 0.) col) in
        let col = List.map (fun x -> if x = 0. then med else x) col in
        let q1 = quantile col 0.25 in
        let q3 = quantile col 0.75 in
        let iqr = q3 -. q1 in
        let lw = q1 -. 1.5 *. iqr in
        let uw = q3 +. 1.5 *. iqr in
        List.map (fun x -> if x < lw then quantile col 0.05 else if x > uw then quantile col 0.95 else x) col
    | 6 -> (* DiabetesPedigreeFunction: cap outliers (IQR) *)
        let q1 = quantile col 0.25 in
        let q3 = quantile col 0.75 in
        let iqr = q3 -. q1 in
        let lw = q1 -. 1.5 *. iqr in
        let uw = q3 +. 1.5 *. iqr in
        List.map (fun x -> if x < lw then quantile col 0.05 else if x > uw then quantile col 0.95 else x) col
    | 7 -> (* Age: cap outliers (IQR) *)
        let q1 = quantile col 0.25 in
        let q3 = quantile col 0.75 in
        let iqr = q3 -. q1 in
        let lw = q1 -. 1.5 *. iqr in
        let uw = q3 +. 1.5 *. iqr in
        List.map (fun x -> if x < lw then quantile col 0.05 else if x > uw then quantile col 0.95 else x) col
    | _ -> col
  ) cols in
  let n = List.length xs in
  let processed = List.init n (fun i -> List.init 8 (fun j -> List.nth (List.nth cols' j) i)) in
  select_best_features processed

let mean_std xs =
  let n = float_of_int (List.length xs) in
  let m = List.fold_left (List.map2 ( +. )) (List.hd xs) (List.tl xs) |> List.map (fun s -> s /. n) in
  let s =
    let sq_diffs = List.map (fun x -> List.map2 (fun xi mi -> (xi -. mi) ** 2.) x m) xs in
    let sum = List.fold_left (List.map2 ( +. )) (List.hd sq_diffs) (List.tl sq_diffs) in
    List.map (fun s -> sqrt (s /. n)) sum
  in
  (m, s)

let zscore_normalize xs (m, s) =
  List.map (fun x -> List.map2 (fun xi (mi, si) -> if si = 0. then 0. else (xi -. mi) /. si) x (List.combine m s)) xs

let rec insert_at n x lst =
  match lst, n with
  | [], _ -> [x]
  | hd :: tl, 0 -> x :: hd :: tl
  | hd :: tl, _ -> hd :: insert_at (n - 1) x tl

let shuffle lst =
  let rec aux acc = function
    | [] -> acc
    | x :: xs ->
        let pos = Random.int (List.length acc + 1) in
        aux (insert_at pos x acc) xs
  in
  aux [] lst


let split_train_test data train_frac =
  let n = List.length data in
  let n_train = int_of_float (train_frac *. float_of_int n) in
  let train = sublist data 0 n_train in
  let test = sublist data n_train (n - n_train) in
  (train, test)

let read_diabetes_csv filename =
  let ic = open_in filename in
  let rec loop acc =
    match input_line ic with
    | line ->
      let values = String.split_on_char ',' line |> List.map String.trim in
      if List.length values = 9 then
        (try
          let floats = List.map float_of_string values in
          let features = sublist floats 0 8 in
          let label = List.nth floats 8 in
          loop ((features, label) :: acc)
        with _ -> loop acc)
      else loop acc
    | exception End_of_file -> close_in ic; List.rev acc
  in
  loop []

let preprocess_dataset xs ys m s =
  let xs = impute_and_cap xs in
  let xs_norm = zscore_normalize xs (m, s) in
  List.combine xs_norm ys

let augment_dataset dataset factor noise_std =
  let perturb x =
    List.map (fun xi -> xi +. (Random.float 2.0 -. 1.0) *. noise_std) x
  in
  (* Separate class 0 and class 1 *)
  let class0 = List.filter (fun (_, y) -> y = 0.) dataset in
  let class1 = List.filter (fun (_, y) -> y = 1.) dataset in

  let replicate data times =
    List.flatten (List.init times (fun _ -> List.map (fun (x, y) -> (perturb x, y)) data))
  in

  let class0_aug = replicate class0 factor in        (* e.g., 10x *)
  let class1_aug = replicate class1 (factor * 3) in  (* e.g., 30x → stronger boost for minority class *)

  let combined = dataset @ class0_aug @ class1_aug in
  shuffle combined

let log_class_distribution dataset =
  let count0 = List.fold_left (fun acc (_, y) -> if y = 0. then acc + 1 else acc) 0 dataset in
  let count1 = List.length dataset - count0 in
  Printf.printf "Class 0 count: %d\n" count0;
  Printf.printf "Class 1 count: %d\n" count1

let load_and_split_diabetes_csv filename =
  let data = read_diabetes_csv filename |> shuffle in
  let train, test = split_train_test data 0.85 in 
  let xs_train, ys_train = List.split train in
  let xs_train = impute_and_cap xs_train in
  let m, s = mean_std xs_train in
  let trainset = List.combine (zscore_normalize xs_train (m, s)) ys_train in
  (* let trainset = augment_dataset trainset 30 0.05 in *)
  (* log_class_distribution trainset;  *)
  let xs_test, ys_test = List.split test in
  let xs_test = impute_and_cap xs_test in
  let testset = List.combine (zscore_normalize xs_test (m, s)) ys_test in
  trainset, testset 