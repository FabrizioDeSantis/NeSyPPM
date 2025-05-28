'''
This file contains the logical rules used for the LTN training.
'''
# SEPSIS KNOWLEDGE BASE
'''
R1: Patients with LacticAcid > 2.0 should be admitted to the ICU. 
    \-/ x (LacticAcid(x) > 2.0 -> ICU(x))
R2: Patients with Tachypnea, Suspected Infection, and Critical Heart Rate, should be admitted to the ICU. 
    \-/ x (Tachypnea(x) /\ SuspectedInfection(x) /\ CriticalHeartRate(x) -> ICU(x))
R3: Patients with CRP activity followed by IV Antibiotics and with CRP value > 100 should be admitted to the ICU. 
    \-/ x (CRP(x) /\ IVAntibiotics(x) /\ CRPValue(x) > 100 -> ICU(x))
R4: If the activity ERTriage occurs, the activity ERSepsisTriage should occur after ERTriage.
R5: If the activity ERSepsisTriage occurs, the activity IV Antibiotics should occur after ERSepsisTriage.
R6: If the activity ERSepsisTriage occurs, the activity IV Liquid should occur after ERSepsisTriage.
'''
# BPI2012 KNOWLEDGE BASE
'''
R1: If the Requested Amount is less than 10000, then the application should not be accepted. 
    \-/ x (RequestedAmount(x) < 10000 -> ApplicationAccepted(x))
R2: If the Requested Amount is greater than 50000 and less than 100000, then the application should not be accepted. 
    \-/ x (RequestedAmount(x) > 50000 /\ RequestedAmount(x) < 100000 -> Not(ApplicationAccepted(x)))
R3: If the resources 10910 and 11169 perform an activity, then the application should not be accepted. 
    \-/ x (HasResource(x,10910) /\ HasResource(x,11169) -> Not(ApplicationAccepted(x)))
R4: If the activity W_Completeren aanvraag-COMPLETE occurs, the activity A_ACCEPTED_COMPLETE should occur after W_Completeren aanvraag-COMPLETE.
R5: If the activity W_Valideren aanvraag-COMPLETE occurs, the activity O_ACCEPTED_COMPLETE should occur after W_Valideren aanvraag-COMPLETE.
R6: If the activity O_SENT_BACK_COMPLETEoccurs, the activity W_Valideren aanvraag-COMPLETE should occur after O_SENT_BACK_COMPLETE.
'''
# BPI2017 KNOWLEDGE BASE
'''
R1: If the Credit Score is greater than 0 and the Requested Amount is less than 20000, then the application should be accepted. 
    \-/ x (CreditScore(x) > 0 /\ RequestedAmount(x) < 20000 -> ApplicationAccepted(x))
R2: If there is no offer with Credit Score greater than 0, then the application should not be accepted. 
    \-/ x (Not(Offer(x)) /\ CreditScore(x) > 0 -> Not(ApplicationAccepted(x)))
R3: If the Requested Amount is greater than 20000 and the Loan Goal is "Existing loan takeover", then the application should be accepted.   
    \-/ x (RequestedAmount(x) > 20000 /\ LoanGoal(x) = "Existing loan takeover" -> ApplicationAccepted(x))
R4: If the activity A_SUBMITTED occurs, the activity A_ACCEPTED should occur after A_SUBMITTED.
R5: If the activity A_ACCEPTED occurs, the activity O_CREATE_OFFER should occur after A_ACCEPTED.
R6: If the activity A_Complete occurs, the activity W_validate application should occur after A_Complete.
'''
# TRAFFIC FINES KNOWLEDGE BASE
'''
R1: If the payment amount is less than the amount of the fine, then the fine shoul be sent for credit collection. 
    \/x (PaymentAmount(x) < FineAmount(x) -> SendToCreditCollection(x))
R2: If the activity Add penalty occurs, then the fine should be sent for credit collection. 
    \/x (HasActivity(x, "Add penalty") -> SendToCreditCollection(x))
R3: If the amount of the fine is greater than 400, then the fine should be sent for credit collection. 
    \/x (FineAmount(x) > 400 -> SendToCreditCollection(x))
R4: If the activity Create Fine occurs, the activity Send Fine should occur after Create Fine.
R5: If the activity Send Fine occurs, the activity Insert Fine Notification should occur after Send Fine.
R6: If the activity Send Fine occurs, the activity Payment Notification should occur after Send Fine.
'''